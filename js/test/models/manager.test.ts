var sinon = require('sinon');
var assert = require('chai').assert;
import {WindowModel} from '../../src/wm/models/WindowModel';
import {WindowManagerModel} from '../../src/wm/models/WindowManagerModel';
import 'mocha';


describe('manager', () => {

    function WindowParams(params) {
        return {
            id: "0",
            x: 0,
            y: 0,
            width: 100,
            height: 100,
            title: 'title',
            ...params
        };
    }

    var manager: any; //WindowManagerModel;

    beforeEach(() => {
        manager = new WindowManagerModel([]);
    });

    describe('.get', () => {

        it('should get a window by its id', () => {
            var win1 = new WindowModel(WindowParams({id: 'a'}));
            var win2 = new WindowModel(WindowParams({id: 'b'}));
            var win3 = new WindowModel(WindowParams({id: 'c'}));

            manager.add(win1);
            manager.add(win2);
            manager.add(win3);

            assert.equal(manager.get('a'), win1);
            assert.equal(manager.get('b'), win2);
            assert.equal(manager.get('c'), win3);
        });

    });

    describe('.has', () => {

        it('should check if a window exists in the manager', () => {
            var window = new WindowModel(WindowParams({id: 0}));

            assert.isFalse(manager.has(window));
            manager.add(window);
            assert.isTrue(manager.has(window));
        });

    });

    describe('.add', () => {

        it('should add a window', () => {
            var window = new WindowModel(WindowParams({id: 'a'}));
            assert.equal(manager.add(window), window);
            assert.equal(manager.length(), 1);
            assert.deepEqual(manager.allWindows(), [window]);
        });

        it('should convert object to window instance', () => {
            var window = {id: 'custom', title: 'my title'};
            window = manager.add(window);
            assert(window instanceof WindowModel);
            assert.equal(manager.length(), 1);
            assert.equal(manager.get('custom'), window);
        });

    });

    describe('.remove', () => {

        it('should remove a window', () => {
            var window = new WindowModel(WindowParams({id: 0}));
            manager.add(window);
            assert.equal(manager.length(), 1);
            manager.remove(window);
            assert.equal(manager.length(), 0);
        });

        it('should remove a window by its id', () => {
            var window = {id: 0};
            manager.add(window);
            assert.equal(manager.length(), 1);
            manager.remove(0);
            assert.equal(manager.length(), 0);
        });

    });

    describe('.open', () => {

        it('should only add a window once', () => {
            var size = 10;
            var component = '<div></div>';
            var props = {id: 20, x: size, y: size, width: size, height: size};

            var window = manager.open(component, props);

            assert(manager.has(window));
            assert.equal(manager.length(), 1);

            assert.equal(manager.open(component, props), window);
            assert.equal(manager.length(), 1);
        });

    });

    describe('.focus', () => {

        it('should increase z-index', () => {
            var win1 = new WindowModel(WindowParams({id: 1}));
            var win2 = new WindowModel(WindowParams({id: 2}));
            var win3 = new WindowModel(WindowParams({id: 3}));

            manager.add(win1);
            manager.add(win2);
            manager.add(win3);
            assert.deepEqual(manager._active, win3);

            manager.focus(win1);
            assert.deepEqual(manager._active, win1);

            manager.focus(win2);
            assert.deepEqual(manager._active, win2);

            manager.focus(win3);
            assert.deepEqual(manager._active, win3);
        });

    });

    describe('.toJSON', () => {

        it('should export to a JS array', () => {
            var props = {
                id: 'my-id',
                x: 200,
                y: 300,
                width: 100,
                height: 50,
                title: 'My amazing window',
                isOpen: true
            };

            var window = manager.add(props);
            assert.deepEqual(manager.toJSON(), [window.toJSON()]);
        });

    });

});
