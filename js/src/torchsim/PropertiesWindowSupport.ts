import {WindowManagerModel} from "../wm/models/WindowManagerModel";
import {WindowModel} from "../wm/models/WindowModel";
import {ObserverManager} from "./ObserverManager";
import {LocalStorageDb} from "./LocalStorageDb";


export const PROPERTIES_PANE_ID = '__properties_pane';

export class PropertiesWindowSupport {
    private windowManager: WindowManagerModel;
    private observerManager: ObserverManager;
    private currentWindowId: string;

    constructor(windowManager: WindowManagerModel, observerManager: ObserverManager) {
        this.windowManager = windowManager;
        this.observerManager = observerManager;

        this.windowManager.signals.changeFocus.connect((window: WindowModel) => {
            if (window.id !== PROPERTIES_PANE_ID && window.propertiesComponent !== null) {
                this.setProperties(window);
            }
        });

        this.windowManager.signals.windowUpdated.connect((window: WindowModel) => {
            if( this.currentWindowId === window.id ) {
                this.setProperties(window);
            }
        });
    }

    addPropertiesPane() {
        if (!this.windowManager.has(PROPERTIES_PANE_ID)) {
            let position = LocalStorageDb.getWindowPosition(PROPERTIES_PANE_ID);
            this.observerManager.createWindow(PROPERTIES_PANE_ID, null, null, position);
            this.windowManager.get(PROPERTIES_PANE_ID).rename("Properties")
        }
    }

    private setProperties(window: WindowModel) {
        this.currentWindowId = window.id;
        this.windowManager.get(PROPERTIES_PANE_ID).rename(`Properties of ${window.id}`);
        this.windowManager.get(PROPERTIES_PANE_ID).setComponent(window.propertiesComponent);
    }
}