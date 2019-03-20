import {WindowPosition} from "./ServerConnector";

export class LocalStorageDb {
    private static getObject(id: string): Object {
        return JSON.parse(localStorage.getItem(id) || '{}');
    }

    static getWindowPosition(win_id: string): WindowPosition | undefined {
        let positions = this.getObject('positions');
        return positions[win_id];
    }

    static setWindowPosition(win_id: string, position: WindowPosition) {
        let positions = JSON.parse(localStorage.getItem('positions') || '{}');
        positions[win_id] = position;
        localStorage.setItem('positions', JSON.stringify(positions));
    }
}