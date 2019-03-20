const assert = require('chai').assert;
const sinon = require('sinon');
import {WindowModel, WindowActionResize} from '../../src/wm/models/WindowModel';

import 'mocha';

describe('window', () => {
    let window: WindowModel;

    beforeEach(() => {
        window = new WindowModel({
            id: "0",
            x: 0,
            y: 0,
            width: 100,
            height: 100,
            title: 'title'
        });
    });

    describe('move', () => {

        it('should move the window to a point', () => {
            // var window = new WindowModel({id: "0"});
            assert.equal(window.x, 0);
            assert.equal(window.y, 0);

            window.startMove(0, 0);

            var x = 200;
            var y = 300;

            window.update(x, y);
            assert.equal(window.x, x);
            assert.equal(window.y, y);

            window.endChange();
        });

    });

    describe('._quadrant', () => {
        let resizeAction;
        beforeEach(() => {
            resizeAction = new WindowActionResize(window, 0, 0);
        });
        it('top left', () => {
            assert.deepEqual(resizeAction._quadrant(1, 1), {
                top: true,
                left: true
            });
        });

        it('top right', () => {
            assert.deepEqual(resizeAction._quadrant(99, 1), {
                top: true,
                left: false
            });
        });

        it('bottom left', () => {
            assert.deepEqual(resizeAction._quadrant(1, 99), {
                top: false,
                left: true
            });
        });

        it('bottom right', () => {
            assert.deepEqual(resizeAction._quadrant(99, 99), {
                top: false,
                left: false
            });
        });

    });

    describe('.resize', () => {
        var start = 100;
        var change = 20;

        beforeEach(() => {
            window.setPosition(start, start);
            window.setSize(start, start);
        });

        afterEach(() => {
            window.endChange();
        });

        describe('left', () => {

            beforeEach(() => {
                window.startResize(start, 0);
            });

            it('in', () => {
                window.update(start - change, 0);
                assert.equal(window.x, start - change);
                assert.equal(window.width, start + change);
            });

            it('out', () => {
                window.update(start + change, 0);
                assert.equal(window.x, start + change);
                assert.equal(window.width, start - change);
            });
        });

        describe('top', () => {

            beforeEach(() => {
                window.startResize(0, start);
            });

            it('in', () => {
                window.update(0, start - change);
                assert.equal(window.y, start - change);
                assert.equal(window.height, start + change);
            });

            it('out', () => {
                window.update(0, start + change);
                assert.equal(window.y, start + change);
                assert.equal(window.height, start - change);
            });
        });

        describe('right', () => {

            beforeEach(() => {
                window.startResize(start * 2, 0);
            });

            it('in', () => {
                window.update(start * 2 - change, 0);
                assert.equal(window.x, start);
                assert.equal(window.width, start - change);
            });

            it('out', () => {
                window.update(start * 2 + change, 0);
                assert.equal(window.x, start);
                assert.equal(window.width, start + change);
            });
        });

        describe('bottom', () => {

            beforeEach(() => {
                window.startResize(0, start * 2);
            });

            it('in', () => {
                window.update(0, start * 2 - change);
                assert.equal(window.y, start);
                assert.equal(window.height, start - change);
            });

            it('out', () => {
                window.update(0, start * 2 + change);
                assert.equal(window.y, start);
                assert.equal(window.height, start + change);
            });
        });

    });

    describe('.open', () => {
    });

    describe('.close', () => {

        it('should close the window', () => {
            assert.equal(window.isOpen, true);

            window.close();
            assert.equal(window.isOpen, false);
        });

    });

    describe('.rename', () => {

        it('should rename the window', () => {
            assert.equal(window.title, 'title');

            var title = 'My custom window title';

            window.rename(title);
            assert.equal(window.title, title);
        });

    });

    describe('.toJSON', () => {

        it('should export to a standard JS object', () => {
            var props = {
                id: 'window-1',
                x: 20,
                y: 30,
                index: 1,
                width: 200,
                height: 400,
                maxWidth: Infinity,
                minWidth: 0,
                maxHeight: Infinity,
                minHeight: 0,
                title: 'Test',
                isOpen: true
            };

            var window = new WindowModel(props);
            assert.deepEqual(window.toJSON(), props);
        });

    });

    describe('.onChange', () => {
        let spy;

        beforeEach(() => {
            spy = sinon.spy();
            window.signals.change.connect(spy);
        });

        it('should trigger on setSize', () => {
            window.setSize(0,0);
            assert(spy.calledOnce);
        });

        it('should trigger on setPosition', () => {
            window.setPosition(0,0);
            assert(spy.calledOnce);
        });

        it('should trigger on move', () => {
            window.startMove(0,0);
            window.update(0, 0);
            window.endChange();
            assert(spy.calledOnce);
        });

        it('should trigger on resize', () => {
            window.startResize(0, 0);
            window.update(0, 0);
            window.endChange();
            assert(spy.calledOnce);
        });

        it('should trigger on open', () => {
            window.open();
            assert(!spy.called);
            window.close();
            assert(spy.calledOnce);
            window.open();
            assert(spy.called);
        });

        it('should trigger on close', () => {
            window.close();
            assert(spy.calledOnce);
            window.close();
            assert(spy.calledOnce);
        });

        it('should trigger on rename', () => {
            window.rename('new name');
            assert(spy.calledOnce);
        });

    });

});
