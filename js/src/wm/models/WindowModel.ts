'use strict';

import {string} from "prop-types";
import {Signal} from "typed-signals";
import {WindowManagerModel} from "./WindowManagerModel";
import {InvalidArgumentException, InvalidStateException} from "../../torchsim/exceptions";

interface WindowModelParams {
    id: string;
    title: string;
    x: number;
    y: number;
    width: number;
    height: number;
    maxWidth?: number;
    minWidth?: number;
    maxHeight?: number;
    minHeight?: number;
}

abstract class WindowAction {
    protected model: WindowModel;

    constructor(model: WindowModel, x: number, y: number) {
        this.model = model;
        this.init(x, y);
    }

    abstract init(x: number, y: number) : void;
    abstract update(x: number, y: number) : boolean;
    abstract finish() : void;
}

class WindowActionInactive extends WindowAction {

    constructor(model: WindowModel) {
        super(model, 0, 0);
    }

    init(x: number, y: number) {
    }
    update(x: number, y: number): boolean {
        return false;
    }
    finish(): void {
    }
}

class WindowActionMove extends WindowAction {
    private _offsetX: number;
    private _offsetY: number;


    init(x: number, y: number) {
        this._offsetX = x - this.model.x;
        this._offsetY = y - this.model.y;
    }
    update(x: number, y: number): boolean {
        this.model.x = x - this._offsetX;
        this.model.y = y - this._offsetY;
        return true;
    }

    finish(): void {
        this.model.signals.changePosition.emit();
        this.model.signals.change.emit();
    }

}

export class WindowActionResize extends WindowAction {
    private _quad: any;
    private _startX: number;
    private _startY: number;
    private _startWidth: number;
    private _startHeight: number;
    private _originX: number;
    private _originY: number;

    init(x: number, y: number) {
        this._quad = this._quadrant(x, y);
        this._startX = this.model.x;
        this._startY = this.model.y;
        this._startWidth = this.model.width;
        this._startHeight = this.model.height;
        this._originX = x;
        this._originY = y;
    }

    update(x: number, y: number): boolean {
        let deltaX = x - this._originX;
        let deltaY = y - this._originY;

        const finalWidth = this._startWidth + (this._quad.left ? deltaX * -1 : deltaX);
        const finalHeight = this._startHeight + (this._quad.top ? deltaY * -1 : deltaY);

        if (finalWidth > this.model.maxWidth) {
            deltaX = this.model.maxWidth - this._startWidth;
            if (this._quad.left) {
                deltaX *= -1;
            }
        } else if (finalWidth < this.model.minWidth) {
            deltaX = this.model.minWidth - this._startWidth;
            if (this._quad.left) {
                deltaX *= -1;
            }
        }

        if (finalHeight > this.model.maxHeight) {
            deltaY = this.model.maxHeight - this._startHeight;
            if (this._quad.top) {
                deltaY *= -1;
            }
        } else if (finalHeight < this.model.minHeight) {
            deltaY = this.model.minHeight - this._startHeight;
            if (this._quad.top) {
                deltaY *= -1;
            }
        }

        if (this._quad.left) {
            this.model.x = this._startX + deltaX;
            this.model.width = this._startWidth - deltaX;
        } else {
            this.model.width = this._startWidth + deltaX;
        }

        if (this._quad.top) {
            this.model.y = this._startY + deltaY;
            this.model.height = this._startHeight - deltaY;
        } else {
            this.model.height = this._startHeight + deltaY;
        }
        return true;
    }

    finish(): void {
        this.model.signals.changeSize.emit();
        this.model.signals.changePosition.emit();
        this.model.signals.change.emit();
    }

    /**
     * Find which quadrant of the window the mouse is
     * @param {number} x: horizontal position of the mouse
     * @param {number} y: vertical position of the mouse
     * @private
     */
    _quadrant(x: number, y: number) {
        return {
            top: y < this.model.y + (this.model.height / 2),
            left: x < this.model.x + (this.model.width / 2)
        };
    }

}

export class WindowModel {
    public signals = {
        change: new Signal<() => void>(),
        changeComponent: new Signal<() => void>(),
        changeIndex: new Signal<() => void>(),
        changePosition: new Signal<() => void>(),
        changeSize: new Signal<() => void>(),
        changeOpen: new Signal<() => void>(),
        changeTitle: new Signal<() => void>(),
    };
    action: WindowAction = new WindowActionInactive(this);
    private manager: WindowManagerModel;

    id: string = undefined;
    x: number = 0;
    y: number = 0;
    index: number = 1;
    width: number = 0;
    height: number = 0;
    maxWidth: number = Infinity;
    minWidth: number = 0;
    maxHeight: number = Infinity;
    minHeight: number = 0;
    title: string = "";
    isOpen: boolean = true;
    component: JSX.Element = undefined;
    propertiesComponent: JSX.Element;


    constructor(params: WindowModelParams) {
        Object.assign(this, params);

        if (this.id === undefined) {
            throw new InvalidArgumentException('All windows must have an id');
        }
    }

    setManager(manager: WindowManagerModel) {
        this.manager = manager;
    }

    setPosition(x: number, y: number) {
        this.x = x;
        this.y = y;
        this.signals.changePosition.emit();
        this.signals.change.emit();
    }

    setSize(width: number, height: number) {
        this.width = width;
        this.height = height;
        this.signals.changeSize.emit();
        this.signals.change.emit();
    }

    /**
     * Set z-index of window
     * @param {number} index
     */
    setIndex(index: number) {
        this.index = index;
        this.signals.changeIndex.emit();
        this.signals.change.emit();
    }

    /**
     * Start moving the window
     * @param {number} x: horizontal position of the mouse
     * @param {number} y: vertical position of the mouse
     */
    startMove(x: number, y: number) {
        this.action = new WindowActionMove(this, x, y);
    }

    /**
     * Start resizing the window
     * @param {number} x: horizontal position of the mouse
     * @param {number} y: vertical position of the mouse
     */
    startResize(x: number, y: number) {
        this.action = new WindowActionResize(this, x, y);
    }

    /**
     * Update a move/resize action
     * @param {number} x: horizontal position of the mouse
     * @param {number} y: vertical position of the mouse
     * @returns {boolean} True when something changed (DOM should be refreshed)s
     */
    update(x: number, y : number) : boolean {
        return this.action.update(x, y);
    }

    /**
     * Finish moving/resizing the window
     */
    endChange() {
        this.action.finish();
        this.action = new WindowActionInactive(this);
    }

    open() {
        if (this.isOpen) {
            return;
        }
        this.isOpen = true;
        this.signals.changeOpen.emit();
        this.signals.change.emit();
    }

    close() {
        if (!this.isOpen) {
            return;
        }
        this.isOpen = false;
        this.signals.changeOpen.emit();
        this.signals.change.emit();
    }

    requestFocus() {
        if (!this.manager) {
            throw new InvalidStateException('Cannot focus a window that is not being managed');
        }
        this.manager.focus(this);
    }

    isFocused() {
        if (!this.manager) {
            return false;
        }
        return this.manager.active() === this;
    }

    /**
     * Rename the window
     * @param {string} title
     */
    rename(title: string) {
        this.title = title;
        this.signals.changeTitle.emit();
        this.signals.change.emit();
    }

    /**
     * Set content of the window
     * @param {JSX.Element} component
     */
    setComponent(component: JSX.Element) {
        this.component = component;
        this.signals.changeComponent.emit();
        this.signals.change.emit();
    }

    /**
     * Export model as JSON
     */
    toJSON() {
        return {
            id: this.id,
            x: this.x,
            y: this.y,
            index: this.index,
            width: this.width,
            height: this.height,
            maxWidth: this.maxWidth,
            minWidth: this.minWidth,
            maxHeight: this.maxHeight,
            minHeight: this.minHeight,
            title: this.title,
            isOpen: this.isOpen
        };
    }

    setPropertiesComponent(properties: JSX.Element) {
        this.propertiesComponent = properties;
    }
}
