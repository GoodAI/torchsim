import {SinonSpyStatic} from "sinon";

const chai = require('chai');
const sinon = require('sinon');
const sinonChai = require('sinon-chai');
import {ElementResizeWatcher} from '../src/wm/GuiUtils';

import 'mocha';

chai.should();
chai.use(sinonChai);

class HTMLElementStub {
    size;

    constructor(size) {
        this.size = size;
    }

    getBoundingClientRect() {
        return this.size;
    }
}

describe('GUI Utils', () => {

    describe('ElementResizeWatcher', () => {

        let createElement = (size) => (<any>new HTMLElementStub(size)) as HTMLElement;

        let watcher: ElementResizeWatcher;
        let resizedSpy;
        beforeEach(() => {
            watcher = new ElementResizeWatcher();
            resizedSpy = sinon.spy();
            watcher.signals.resized.connect(resizedSpy);

        });

        it('should call resized after element is set', () => {
            let size = {width: 100, height: 200};
            watcher.setElement(createElement(size));
            resizedSpy.should.have.been.calledOnceWithExactly(size);
        });

        it('should call resized after update when size is changed', () => {
            let size = {width: 100, height: 200};
            let size2 = {width: 120, height: 200};
            const element = new HTMLElementStub(size);
            watcher.setElement(<any>element as HTMLElement);
            element.size = size2;
            watcher.update();
            resizedSpy.should.have.been.calledTwice;
            resizedSpy.should.have.been.calledWithExactly(size);
            resizedSpy.should.have.been.calledWithExactly(size2);
        });

        it('should not call resized  after update when size is not changed', () => {
            let size = {width: 100, height: 200};
            watcher.setElement(createElement(size));
            watcher.update();
            resizedSpy.should.have.been.calledOnceWithExactly(size);
        });

    });
});
