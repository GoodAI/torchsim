import {Signal} from "typed-signals";
import * as request from 'superagent';
import {InvalidStateException} from "./exceptions";


export enum CommandType {
    Window = "window",
    RequestResponse = "request_response",
    Close = "close"
}

export interface WindowPosition {
    x: number,
    y: number,
    width: number,
    height: number
}

export interface Command {
    cmd: CommandType,
    data: CommandWindow | CommandClose,
    requestId: any // TODO move this to RequestResponse type
}

export interface CommandClose {
    id: string // "Log"
}

export interface CommandWindow {
    id: string, // "Log"
    type: string, // "text"
    content: any, // "App started<br/>"
    position?: WindowPosition

    // command: string, // "window"
    // command: string, // "window"
    // content: any, // "App started<br/>"
    // contentID: string, // "369ee41c5113c0"
    // height: number, // null
    // i: number, // 0
    // id: string, // "Log"
    // inflate: boolean, // true
    // title: string, // "Text Log"
    // type: string, // "text"
    // width: number // null
}

/**
 * Server connector - connecting to UI server.
 *
 * Initial received commands:
 * 1. window (multiple times)
 *
 * Save layout commands:
 * -> save_layout
 */
export class ServerConnector {
    public signals = {
        connectedChange: new Signal<(connected: boolean) => void>(),
        receivedWindow: new Signal<(command: CommandWindow) => void>(),
        receivedLayoutUpdate: new Signal<(data: string) => void>(),
        receivedClose: new Signal<(windowId: string) => void>(),
    };
    private _socket;
    private sessionID;
    readonly: boolean;
    private readonly envId;
    private requestQueue : RequestQueue;


    constructor(envId) {
        this.envId = envId;
        this.requestQueue = new RequestQueue(this.sendSocketMessage)
    }

    connect() {
        if (this._socket) {
            return;
        }
        const url = window.location;
        let ws_protocol = null;
        if (url.protocol === "https:") {
            ws_protocol = 'wss';
        } else {
            ws_protocol = 'ws';
        }
        const socket = new WebSocket(ws_protocol + '://' + url.host + ServerConnector.correctPathname() + 'socket');

        socket.onmessage = (msg) => this.handleMessage(msg);
        socket.onopen = () => this.signals.connectedChange.emit(true);
        socket.onerror = socket.onclose = () => {
            this.signals.connectedChange.emit(false);
            this._socket = null;
        };

        this._socket = socket;
    };

    private handleMessage(msg) {
        const cmd: Command = JSON.parse(msg.data);

        switch (cmd.cmd) {
            // case 'register':
            //     this.sessionID = cmd.data;
            //     this.readonly = cmd.readonly; // Not used
            //     this.postForEnv(this.envId);
            //     break;
            case 'window':
                this.signals.receivedWindow.emit(cmd.data as CommandWindow);
                break;
            // case 'reload':
            //     // TODO implement reload
            //     // for (var it in cmd.data) {
            //     //     localStorage.setItem(this.keyLS(it), JSON.stringify(cmd.data[it]));
            //     // }
            //     break;
            case 'close':
                this.signals.receivedClose.emit((cmd.data as CommandClose).id);
                break;
            // case 'layout':
            //     // this.relayout();
            //     // this.loadLayoutById(DEFAULT_LAYOUT);
            //     break;
            // case 'env_update':
            //     // Multiple environments are not supported
            //     break;
            // case 'layout_update':
            //     this.signals.receivedLayoutUpdate.emit(cmd.data);
            //     break;
            case 'request_response':
                this.requestQueue.resolve(cmd.requestId, cmd.data);
                break;
            default:
                console.error('unrecognized command', cmd);
        }
    };

    private static correctPathname() {
        let pathname = window.location.pathname;
        if (pathname.indexOf('/env/') > -1) {
            pathname = pathname.split('/env/')[0];
        } else if (pathname.indexOf('/compare/') > -1) {
            pathname = pathname.split('/compare/')[0];
        }
        if (pathname.slice(-1) != '/') {
            pathname = pathname + '/';
        }
        return pathname;
    }

    private postForEnv(envId) {
        // console.log("postForEnv");
        request
            .post(`${ServerConnector.correctPathname()}env/${envId}`)
            .send({'sid': this.sessionID}) // sends a JSON post body
            // .set('X-API-Key', 'foobar')
            .set('accept', 'json')
            .end((err, res) => {
                // Calling the end function will send the request
                console.log("result", err, res);
            });

        // Environment comparing endpoint not used: this.correctPathname() + 'compare/' + envIds.join('+')
    };

    sendSocketMessage = (data) => {
        if (!this._socket) {
            throw new InvalidStateException("Cannot send data in not connected state")
        }
        let msg = JSON.stringify(data);
        return this._socket.send(msg);
    }

    sendSocketRequest<T>(data: any) : Promise<T> {
        return new Promise<T>((resolve, reject) => this.requestQueue.createPromise(resolve, reject, data));
    }


}

interface PromiseFunctions<T> {
    resolve: (T) => void;
    reject: () => void;
}

class RequestQueue {
    private sendSocketMessage: (data) => any;

    constructor(sendSocketMessage: (data) => any) {
        this.sendSocketMessage = sendSocketMessage;
    }

    private promises : Dictionary<PromiseFunctions<any>> = {};
    private lastId: number = 0;

    private generateId() : string {
        this.lastId++;
        return `r${this.lastId}`;
    }

    createPromise(resolve, reject, data) {
        let id = this.generateId();
        this.promises[id] = {
            resolve: resolve,
            reject: reject
        };

        let dataFinal = Object.assign({
            cmd: 'request',
            requestId: id,
        }, data);
        this.sendSocketMessage(dataFinal)
    }

    resolve(id, data) {
        let promise = this.promises[id];
        if( promise != undefined ) {
            this.promises[id].resolve(data);
            delete this.promises[id];
        } else {
            console.error(`Received response to not registered promise '${id}'`);
        }
    }

    reject(id) {
        let promise = this.promises[id];
        if( promise != undefined ) {
            this.promises[id].reject();
            delete this.promises[id];
        } else {
            console.error(`Reject called on not registered promise '${id}'`);
        }
    }

}