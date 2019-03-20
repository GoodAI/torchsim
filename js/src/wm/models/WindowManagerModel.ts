'use strict';

import {WindowModel} from "./WindowModel";
import {Signal} from "typed-signals";

var _ = require('lodash');

var INITIAL_INDEX = 1;
var MAX_INDEX = 1000;

export class WindowManagerModel {
    public signals = {
        add: new Signal<(WindowModel) => void>(),
        remove: new Signal<(WindowModel) => void>(),
        change: new Signal<() => void>(),
        changeWindows: new Signal<() => void>(),
        windowPositionChanged: new Signal<(WindowModel) => void>(),
        changeFocus: new Signal<(WindowModel) => void>(),
        changeWindowOpen: new Signal<(WindowModel) => void>(),
        windowUpdated: new Signal<(WindowModel) => void>(),
    };
    private _windows = {};
    private _index = INITIAL_INDEX;
    private _active = undefined;

    constructor(windows: WindowModel[]) {
        windows.forEach(this.add, this);
        // if (_.isArray(windows)) {
        //     windows.forEach(this.add, this);
        // }

        this._resetIndex();
    }

    /**
     * Get a window by it's id
     * @param {string} id
     * @returns {WindowModel}
     */
    get(id: string): WindowModel {
        return this._windows[id];
    }


    /**
     * Check if a window exists in this manager
     * @param {WindowModel | string} window
     * @returns {boolean}
     */
    has(window: WindowModel | string) {
        let id = _.isObject(window) ? (window as WindowModel).id : window as string;
        return this._windows.hasOwnProperty(id);
    }

    /**
     * Add a window
     * @param {WindowModel} window
     * @returns {WindowModel}
     */
    add(window: WindowModel) {
        // if (!(window instanceof WindowModel)) {
        //     window = new WindowModel(window);
        // }
        window.setManager(this);

        this._windows[window.id] = window;
        // this.focus(window);

        window.signals.changeOpen.connect(() => this.signals.change.emit());
        window.signals.change.connect(() => this.signals.changeWindows.emit());
        window.signals.changePosition.connect(() => this.signals.windowPositionChanged.emit(window));

        this.signals.add.emit(window);
        this.signals.change.emit();

        return window;
    }

    /**
     * Remove a window
     * @param {WindowModel} window
     * @returns {WindowModel}
     */
    remove(window: WindowModel | string) {
        let id = _.isObject(window) ? (window as WindowModel).id : window as string;
        window = this.get(id);

        if (!window) {
            throw new Error('Can not a window that it cannot find: ' + id);
        }

        delete this._windows[id];

        this.signals.remove.emit(window);
        this.signals.change.emit();

        return window;
    }

    /**
     * Open a window
     * @param {string} id
     * @param {React.Component} component
     * @param {object} defaults
     * @returns {WindowModel}
     */
    open(id: string, component: JSX.Element, properties: JSX.Element, defaults: any) {
        if (!defaults) {
            defaults = {};
        }
        defaults.id = id;

        const windowExists = this.has(id);
        const window = windowExists ? this.get(id) : this.add(new WindowModel(defaults));
        window.setComponent(component);
        window.setPropertiesComponent(properties);
        window.open();
        if (!windowExists) {
            this.focus(window);
            window.signals.changeOpen.connect(() => {this.signals.changeWindowOpen.emit(window)});
        } else {
            this.signals.windowUpdated.emit(window);
        }
        // this.signals.change.emit();
        return window;
    }

    /**
     * Count how many windows are open
     * @returns {number}
     */
    length(): number {
        return _.keys(this._windows).length;
    }

    /**
     * Focus a window
     * @param {WindowModel | string} id
     */
    focus(id: WindowModel | string) {
        let window = _.isObject(id) ? id as WindowModel : this.get(id as string);


        if (!window) {
            throw new Error('Can not focus on a window it cannot find: ' + id);
        } else if (window === this._active) {
            // this window already has focus
            return;
        }

        if (this._index >= MAX_INDEX) {
            this._resetIndex();
        }

        window.setIndex(this._index);
        this._index += 1;
        this._active = window;
        this.signals.change.emit();
        this.signals.changeFocus.emit(window);
    }

    /**
     * Get active window
     * @returns {boolean}
     */
    active() {
        // TODO check
        return this._active;
    }

    /**
     * Get all windows (open and closed)
     */
    allWindows() {
        return _.values(this._windows);
    }

    /**
     * Get all open windows
     * @returns {any}
     */
    openWindows() {
        return this.allWindows().filter(function (window) {
            return window.isOpen;
        });
    }

    toJSON() {
        return this.allWindows().map(function (window) {
            return window.toJSON();
        });
    }

    toString() {
        return JSON.stringify(this);
    }


    /**
     * Reset window index to 0
     * @private
     */
    private _resetIndex() {
        this._index = INITIAL_INDEX;
        _.sortBy(this.allWindows(), 'index').forEach(function (window) {
            window.setIndex(this._index);
            this._index += 1;
            this._active = window;
        }, this);
    }
}
