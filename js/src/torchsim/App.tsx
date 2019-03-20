import * as React from "react";

import {WindowManagerModel} from "../wm/models/WindowManagerModel";
import {Desktop} from '../wm/views/Desktop'
import {ServerConnector} from "./ServerConnector";
import * as Button from 'react-bootstrap/lib/Button';
import * as classNames from 'classnames';
import {PropertiesWindowSupport, PROPERTIES_PANE_ID} from "./PropertiesWindowSupport";
import {ObserverManager} from "./ObserverManager";
import {WindowModel} from "../wm/models/WindowModel";
import {LocalStorageDb} from "./LocalStorageDb";

export class AppApi {
    private readonly app: App;
    private readonly windowId: string;

    constructor(app: App, windowId: string) {
        this.app = app;
        this.windowId = windowId;

    }

    sendPaneMessage(data) {
        this.app.sendWindowEvent(this.windowId, data);
    }

    sendPaneRequest<T>(requestName: string, data: object) :  Promise<T> {
        return this.app.sendPaneRequest(this.windowId, requestName, data);
    }
}

const DEFAULT_ENV = 'main';

// [ms] Delay UI refresh by the value. All refresh requests received during the period are merged to one.
export const DEFAULT_UI_BATCH_DELAY = 100;

export class App extends React.Component {
    state = {
        envId: DEFAULT_ENV,
        layoutId: 'l1', //'current',
        winPositions: {},
        sessionID: '',
        connected: false
    };

    manager = new WindowManagerModel([]);

    private readonly serverConnector = new ServerConnector(DEFAULT_ENV);
    private readonly propertiesWindowSupport: PropertiesWindowSupport;
    private readonly observerManager: ObserverManager;

    constructor(props) {
        super(props);

        this.observerManager = new ObserverManager(this.manager, (windowId) => new AppApi(this, windowId));
        this.propertiesWindowSupport = new PropertiesWindowSupport(this.manager, this.observerManager);

        // Window manager handlers
        this.manager.signals.windowPositionChanged.connect((window: WindowModel) => {
            // this.exportLayoutsToServer();
            this.sendWindowPositionToServer(window);

            // Store positions to local storage
            if(window.id == PROPERTIES_PANE_ID) {
                LocalStorageDb.setWindowPosition(window.id, {
                    x: window.x,
                    y: window.y,
                    width: window.width,
                    height: window.height
                });
            }
        });

        // Server connectors handlers
        this.serverConnector.signals.receivedWindow.connect((cmd) => {
            this.observerManager.addWindow(cmd);
        });
        this.serverConnector.signals.receivedClose.connect((windowId) => {
            this.closeWindow(windowId);
        });
        // this.serverConnector.signals.receivedLayoutUpdate.connect((data) => {
        //     this.parseLayoutsFromServer(JSON.parse(data));
        // });
        this.serverConnector.signals.connectedChange.connect((connected) => {
            this.setState({connected: connected});
            if(connected) {
                this.propertiesWindowSupport.addPropertiesPane();
            }
        });

        this.manager.signals.changeWindowOpen.connect( (window:WindowModel) => {
            if( !window.isOpen) {
                this.sendWindowEvent(
                    window.id,
                    {
                        event_type: 'window_closed'
                    }
                )
            }
        })
    }

    componentDidMount() {
        // window.addEventListener("resize", this.updateDimensions);
        // this.updateDimensions();
        this.serverConnector.connect();
    }

    sendWindowPositionToServer(window: WindowModel) {

        this.serverConnector.sendSocketMessage({
            cmd: 'update_window_position',
            data: {
                id: window.id,
                x: window.x,
                y: window.y,
                width: window.width,
                height: window.height
            }
        });
    }
    // exportLayoutsToServer() {
    //     const json = this.manager.toJSON();
    //     const data = json.reduce((acc, item) => ({...acc, [item.id]: item}), {});
    //     let exportForm = JSON.stringify({
    //         [this.state.envId]: {
    //             [this.state.layoutId]: data
    //         }
    //     });
    //     this.serverConnector.sendSocketMessage({
    //         cmd: 'save_layouts',
    //         data: exportForm,
    //     });
    // }

    // parseLayoutsFromServer(layouts) {
    //     const layout = layouts[this.state.envId][this.state.layoutId];
    //     this.observerManager.setLayout(layout);
    //     this.propertiesWindowSupport.addPropertiesPane();
    // }


    closeWindow(windowId: string) {
        if (!this.manager.has(windowId)) {
            console.error(`WARNING: Close pane called on non existing window {windowId}`);
            return;
        }
        this.manager.remove(windowId);
    }


    /**
     * Send message to backend.
     *
     * The `data` object is extended by pane and environment Id.
     * This function is exposed to Pane components through `appApi` prop.
     * Note: Only focused panes should call this method.
     *
     * @param windowId
     * @param data Data to be sent to backend.
     */
    sendWindowEvent(windowId: string, data: object) {
        let dataFinal = Object.assign({
            // eid: this.state.envId,
            win_id: windowId
        }, data);
        // console.log("Send", data, dataFinal);
        this.serverConnector.sendSocketMessage({
            cmd: 'event',
            data: dataFinal
        });
    }

    sendPaneRequest<T>(windowId: string, requestName: string, data: any): Promise<T> {
        let dataFinal = {
            // eid: this.state.envId,
            win_id: windowId,
            request_type: requestName,
            data: data
        };
        // console.log("Send Request", data, dataFinal);
        return this.serverConnector.sendSocketRequest({
            cmd: 'request',
            data: dataFinal
        });
    }


    onReset = () => {
        let exportForm = JSON.stringify({
            [this.state.envId]: {
                [this.state.layoutId]: {}
            }
        });
        this.serverConnector.sendSocketMessage({
            cmd: 'save_layouts',
            data: exportForm,
        });
        location.reload();
    };

    render() {
        // const connected = this.state.connected
        // ?

        var classes = classNames('pull-right', 'label', {
            'label-danger': !this.state.connected,
            'label-success': this.state.connected
        });
        return ([
            <div id="header" className="clearfix">
                <h4 className={classes}>{this.state.connected ? 'online' : 'offline'}</h4>
                <Button onClick={this.onReset} className="pull-right">Reset</Button>
                <div className="text pull-left">TorchSim</div>
            </div>,
            <Desktop manager={this.manager}/>
        ])
    }
}
