import {AppApi} from "../App";
import {ElementResizeWatcher, Size} from "../../wm/GuiUtils";

export interface ObserverProps {
    id: string; // Window id
    name: string; // Observer name
    onFocus: (string, VoidCallback) => void;
    onResize: (rect: Size) => void;
    appApi: AppApi,
}

export class ObserverUtils {
    component: React.Component<ObserverProps, {}>;

    resizeWatcher = new ElementResizeWatcher();


    constructor(component: React.Component<ObserverProps, {}>) {
        this.component = component;

        this.resizeWatcher.signals.resized.connect((rect) => {
            if( component.props.onResize !== undefined) {
                component.props.onResize(rect);
            }
        })

    }

    onRender() {
        this.resizeWatcher.update();
    }

    setContentRef(element: HTMLElement) {
        this.resizeWatcher.setElement(element);
    }
}