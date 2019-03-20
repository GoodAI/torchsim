import {Signal} from "typed-signals";

export interface Size {
    width: number,
    height: number
}

export class ElementResizeWatcher {
    signals = {
        resized: new Signal<(Size) => void>()
    };

    lastSize: Size = undefined;
    private element: HTMLElement = undefined;

    setElement(element: HTMLElement) {
        this.element = element;
        this.checkAndEmit();
    }

    update() {
        this.checkAndEmit();
    }

    private checkAndEmit() {
        if (this.element === undefined || this.element === null) {
            return;
        }
        const size = this.getElementSize();
        const isSame = this.lastSize !== undefined &&
            (size.width === this.lastSize.width &&
                size.height === this.lastSize.height);
        this.lastSize = size;
        if (!isSame) {
            // console.log("Size", this.element.scrollWidth, this.element.clientWidth);
            this.signals.resized.emit(size);
        }
    }

    private getElementSize(): Size {
        const rect = <DOMRect>this.element.getBoundingClientRect();
        const hasVerticalScrollBar = this.element.scrollWidth > this.element.clientWidth;
        const hasHorizontalScrollBar = this.element.scrollWidth > this.element.clientWidth;
        const scrollBarSize = 200;  // correct value is 18
        return {
            width: rect.width + (hasVerticalScrollBar ? scrollBarSize : 0),
            height: rect.height + (hasHorizontalScrollBar ? scrollBarSize : 0)
        }
        // const parent = this.element.parentElement;
        // return {
        //     width: parent.offsetWidth,
        //     height: parent.offsetHeight
        // }
    }
}

export class BatchedExecution {
    private readonly action: () => void;
    private readonly delay: number;
    private timeoutTriggered: boolean = false;


    constructor(action: () => void, delay: number) {
        this.action = action;
        this.delay = delay;
    }

    run() {
        if( this.timeoutTriggered ) return;
        this.timeoutTriggered = true;
        setTimeout(() => {
            this.timeoutTriggered = false;
            this.action();
        }, this.delay);
    }
}