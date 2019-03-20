import traceback
import logging
from typing import List, Callable

from torchsim.gui.server.ui_api import UIApi
from torchsim.gui.observables import ObserverPropertiesItemType, ObserverPropertiesItemSourceType
from torchsim.gui.observer_system import ObserverPropertiesItem, ObserverPersistence
from torchsim.gui.server.ui_server_connector import EventData, RequestData, EventDataPropertyUpdated

logger = logging.getLogger(__name__)


class PropertiesHolder:
    @property
    def properties(self) -> List[ObserverPropertiesItem]:
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = value

    def __init__(self, properties: List[ObserverPropertiesItem]):
        self._properties = properties


class UIHelper:
    _observer_persistence: ObserverPersistence
    _ui_api: UIApi

    def __init__(self, ui_api: UIApi, observer_persistence: ObserverPersistence):
        self._observer_persistence = observer_persistence
        self._ui_api = ui_api
        self._properties = {}

    @property
    def ui_api(self) -> UIApi:
        return self._ui_api

    def properties(self, win: str, items: List[ObserverPropertiesItem]):
        if win not in self._properties:
            self._properties[win] = PropertiesHolder(items)
        else:
            self._properties[win].properties = items

        properties = self.convert_properties(items)
        self._ui_api.properties(properties, win=win)
        self.clear_callbacks(win)
        self.register_properties_callback(win, self._properties[win])

    def clear_callbacks(self, win: str):
        self._ui_api.clear_event_handlers(win)
        self._ui_api.clear_request_handlers(win)

    # def register_callback(self, win: str, command: str, handler: Callable[[Dict[str, any]], None]):
    #     def cb(message: Dict[str, any]):
    #         if message['cmd'] == command:
    #             try:
    #                 if command == 'request':
    #                     result = handler(message['data'])
    #                     self._ui_api.send_request_response(message['requestId'], result)
    #                 else:
    #                     handler(message)
    #             except Exception as e:
    #                 logger.error(last_exception_as_html())
    #
    #     self._ui_api.register_event_handler(cb, win)

    def register_event_callback(self, win_id: str, handler: [[EventData], None]):
        self.ui_api.register_event_handler(win_id, handler)

    def register_request_callback(self, win_id: str, handler: [[RequestData], None]):
        self.ui_api.register_request_handler(win_id, handler)

    def get_properties_callback(self, win: str, properties_holder: PropertiesHolder, no_update: bool = False) -> Callable[[EventData], None]:
        def properties_callback(event: EventDataPropertyUpdated):
            properties = self.convert_properties(properties_holder.properties)

            if event.event_type == 'property_updated':
                prop_id = event.property_id
                value = event.value

                item = properties_holder.properties[prop_id]
                if item.callback is not None:
                    # logger.info(f'Properties {item.name}: {item.value}, new value={value}')
                    # noinspection PyBroadException
                    try:
                        new_value = item.callback(value)
                        # logger.info(f'Callback value: {new_value}')

                        # Convert value to string so it is compatible with UI
                        if self._observer_persistence is not None and \
                                item.source_type != ObserverPropertiesItemSourceType.MODEL:
                            self._observer_persistence.store_value(win, item.name, new_value)
                        # if new_value is not None:
                        #     properties[prop_id]['value'] = new_value
                        #     properties_holder.properties[prop_id].value = new_value
                        #     for p in properties:
                        #         logger.info(f'P {p["name"]}: {p["value"]}')
                        #     if not no_update:
                        #         self._ui_api.properties(properties, win=win)
                    except Exception as e:
                        print(f'Exception in UI callback:')
                        traceback.print_exc()

        return properties_callback

    def register_properties_callback(self, win, properties_holder: PropertiesHolder, no_update: bool = False):
        self.ui_api.register_event_handler(
            win,
            self.get_properties_callback(win, properties_holder, no_update)
        )
        # self.register_callback(win, 'event', self.get_properties_callback(win, properties_holder, no_update))

    @staticmethod
    def convert_properties(items: List[ObserverPropertiesItem]):
        def convert_property(prop_id: int, item: ObserverPropertiesItem):
            result = {
                'id': prop_id,
                'type': ObserverPropertiesItem.TYPE_MAPPING[item.type],
                'name': item.name,
                'value': item.value,
                'state': item.STATE_MAPPING[item.state],
                'optional': item.optional,
                'hint': item.hint,
                'sourceType': ObserverPropertiesItem.SOURCE_TYPE_MAPPING[item.source_type]
            }
            if item.type == ObserverPropertiesItemType.SELECT:
                result['values'] = [i.name for i in item.select_values]
            return result

        return [convert_property(prop_id, item) for prop_id, item in enumerate(items)]
