import pytest

from torchsim.gui.server.ui_server_connector import EventParser, EventDataPropertyUpdated, EventData


class TestEventParser:
    # noinspection PyArgumentList
    @pytest.mark.parametrize('packet, expected_object', [
        ({
            'win_id': 'win1',
            'event_type': 'property_updated',
            'property_id': 1,
            'value': 'val1',
        }, EventDataPropertyUpdated('win1', 'property_updated', 1, 'val1')),
        ({
            'win_id': 'win1',
            'event_type': 'window_closed',
        }, EventData('win1', 'window_closed'))
    ])
    def test_parse_property_updated(self, packet, expected_object):
        assert expected_object == EventParser.parse(packet)
