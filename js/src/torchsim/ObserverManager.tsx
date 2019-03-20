import * as React from "react";

import {WindowManagerModel} from "../wm/models/WindowManagerModel";
import {TextObserver} from "./observers/TextObserver";
import {ImageObserver} from "./observers/ImageObserver";
import {PropertiesObserver} from "./observers/PropertiesObserver";
import {MemoryBlockObserver} from "./observers/MemoryBlockObserver";
import {HierarchicalObserver} from "./observers/HierarchicalObserver";
import {CommandWindow, WindowPosition} from "./ServerConnector";
import {PropertiesTable} from "./components/PropertiesTable";
import {Size} from "../wm/GuiUtils";
import {AppApi} from "./App";
import {ClusterObserver} from "./observers/ClusterObserver";


type AppApiCreator = (windowId: string) => AppApi;

export class ObserverManager {
    private readonly manager: WindowManagerModel;
    private layout: any;
    private readonly appApiCreator: AppApiCreator;

    constructor(windowManager: WindowManagerModel, appApiCreator: AppApiCreator) {
        this.manager = windowManager;
        this.appApiCreator = appApiCreator;
    }

    setLayout(layout) {
        this.layout = layout;
        // TODO: update windows position (now is not needed - layout is received before windows)
    }

    addWindow(cmd: CommandWindow) {
        const winId = cmd.id;
        const content = this.getWindowContent(cmd);
        const properties = this.getWindowProperties(cmd);
        const position = cmd.position;
        this.createWindow(winId, content, properties, position);
    }

    createWindow(winId: string, content: JSX.Element, properties: JSX.Element, position?: WindowPosition) {
        // console.log("Add pane", winId);
        // const window = this.layout != undefined ? this.layout[winId] : undefined;

        // Determine position
        // let position;
        // if (window !== undefined) {
        //     // console.log("Adding existing pane", winId, window);
        //     position = {
        //         x: window.x,
        //         y: window.y,
        //         width: window.width,
        //         height: window.height
        //     };
        // } else {
        //     // console.log("Adding new pane", cmd.data);
        //     position = this.getNewWindowPosition();
        // }

        position = position || this.getNewWindowPosition();
        let params = Object.assign({title: winId}, position);
        // console.log("Content", content);
        this.manager.open(winId, content, properties, params);
    }

    getNewWindowPosition() {
        let count = this.manager.length();
        return {
            width: 350,
            height: 100,
            x: count * 10,
            y: count * 10
        }
    }

    getWindowContent(command: CommandWindow): JSX.Element {
        const observers = {
            memory_block: MemoryBlockObserver,
            image: ImageObserver,
            // plot: PlotPane,
            text: TextObserver,
            properties: PropertiesObserver,
            hierarchical: HierarchicalObserver,
            cluster_observer: ClusterObserver,
        };
        const Observer = observers[command.type];

        return Observer !== undefined ?
            <Observer
                name={command.id}
                data={command.content}
                onFocus={() => console.log("onFocus")}
                appApi={this.appApiCreator(command.id)}
                onResize={(rect) => this.onWindowResized(command.id, rect)}
            />
            : <div>{JSON.stringify(command)}</div>;
    }

    getWindowProperties(command: CommandWindow): JSX.Element {
        const windowsWithProperties = ['memory_block', 'hierarchical', 'cluster_observer'];
        if (windowsWithProperties.includes(command.type)) {
            const content = command.content;
            return <PropertiesTable
                data={content.properties}
                appApi={this.appApiCreator(command.id)}
            />
        } else {
            return null;
        }
    }

    onWindowResized = (winId: string, rect: Size) => {
        const windowBorder = {
            width: 8,  // 8 is border
            height: 25  // 35 is title bar and border
        };
        const newSize = {
            width: rect.width + windowBorder.width,
            height: rect.height + windowBorder.height
        };
        setTimeout(() => {
            this.manager.get(winId).setSize(newSize.width, newSize.height);

        }, 10);
    };

}