'use strict';
import * as React from "react";
import * as ReactDOM from 'react-dom';
import {Offset} from "./Desktop";
import {WindowModel} from "../models/WindowModel";
import {SignalConnection} from "typed-signals/src/Signal";
import * as Alert from 'react-bootstrap/lib/Alert';
import * as Tooltip from 'react-bootstrap/lib/Tooltip';
import * as Popover from 'react-bootstrap/lib/Popover';
import * as OverlayTrigger from 'react-bootstrap/lib/OverlayTrigger';
import * as Overlay from 'react-bootstrap/lib/Overlay';
import * as Glyphicon from 'react-bootstrap/lib/Glyphicon';
import * as Dropdown from 'react-bootstrap/lib/Dropdown';
import * as DropdownButton from 'react-bootstrap/lib/DropdownButton';
import * as MenuItem from 'react-bootstrap/lib/MenuItem';

import * as classNames from 'classnames';
import {PopupMenu} from "../../torchsim/components/PopupMenu";
import {PopupMenuToggle} from "../../torchsim/components/PopupMenuToggle";
import {BatchedExecution} from "../GuiUtils";
import {DEFAULT_UI_BATCH_DELAY} from "../../torchsim/App";

export interface WindowProps {
    window: WindowModel;
    offset: Offset
}

interface WindowState {
    hasError: boolean;
    error: Error;
    showProperties: boolean;
}

export class Window extends React.Component<WindowProps, {}> {
    private readonly window: WindowModel;
    private windowChangeSignal: SignalConnection;
    private ref: HTMLElement;
    private batchedForceUpdate: BatchedExecution =
        new BatchedExecution(() => this.forceUpdate(), DEFAULT_UI_BATCH_DELAY);

    state: WindowState = {
        hasError: false,
        error: null,
        showProperties: false
    }

    constructor(props: WindowProps) {
        super(props);
        this.window = props.window;
    }

    componentDidMount() {
        this.windowChangeSignal = this.window.signals.change.connect(() => this.batchedForceUpdate.run());
        document.addEventListener('mousemove', this.handleMouseMove);
        document.addEventListener('mouseup', this.handleMouseUp);
    }

    componentWillUnmount() {
        this.windowChangeSignal.disconnect();
        document.removeEventListener('mousemove', this.handleMouseMove);
        document.removeEventListener('mouseup', this.handleMouseUp);
    }

    quickUpdate = () => {
        requestAnimationFrame(() => {
            let el = ReactDOM.findDOMNode(this) as HTMLElement;
            el.style.width = this.window.width + 'px';
            el.style.height = this.window.height + 'px';
            el.style.top = this.window.y + 'px';
            el.style.left = this.window.x + 'px';
        });
    };

    preventDefault = (e) => {
        e.preventDefault();
        return false;
    }

    handlePropagation = (e) => {
        if (!(e.ctrlKey || e.metaKey || e.altKey || e.button !== 0)) {
            this.focus();
            e.stopPropagation();
        }
    }

    handleResize = (e) => {
        this.focus();
        var mouse = this.convertPoints(e);
        this.window.startResize(mouse.x, mouse.y);
        e.stopPropagation();
    }

    handleMove = (e, isHeader: boolean) => {
        if( !(e.altKey || isHeader)) {
            return;
        }
        switch (e.button) {
            case 0: // Left mouse button
                // Move
                e.preventDefault();
                this.focus();
                var mouse = this.convertPoints(e);
                this.window.startMove(mouse.x, mouse.y);
                break;
            case 2: // Right mouse button
                // resize
                this.handleResize(e);
        }
    }

    handleMouseMove = (e) => {
        const mouse = this.convertPoints(e);
        const refresh = this.window.update(mouse.x, mouse.y);
        if (refresh) {
            this.quickUpdate();
        }
    };

    handleMouseUp = () => {
        this.window.endChange();
    }

    focus = () => {
        this.window.requestFocus();
    }

    close = () => {
        this.window.requestFocus();
        this.window.close();
    }

    toggleProperties = () => {
        this.setState({showProperties: !this.state.showProperties});
    };

    convertPoints(e) {
        return {
            x: e.clientX - this.props.offset.left,
            y: e.clientY - this.props.offset.top
        };
    }

    componentDidCatch(error, info) {
        // You can also log the error to an error reporting service
        console.error("ERROR", error, info);
        this.setState({
            hasError: true,
            error: error
        });
    }

    render() {
        var classes = classNames({
            window: true,
            active: this.window.isFocused()
        });

        var styles = {
            top: this.window.y,
            left: this.window.x,
            width: this.window.width,
            height: this.window.height,
            zIndex: this.window.index
        };

        const content = this.state.hasError
            ? (<div>
                <Alert bsStyle="danger">
                    <h4>Exception thrown:</h4>
                    {this.state.error.name}: {this.state.error.message}
                    <pre>
                    {this.state.error.stack}
                    </pre>
                </Alert>
            </div>)
            : this.window.component;


        // const propertiesContent = this.window.propertiesComponent !== null
        //     ? (<Popover id="popover-positioned-top" title="Properties">
        //             {this.window.propertiesComponent}
        //         </Popover>
        //     ) : null;

        // Menu is prepared, but not used
        const menu = (
            <Dropdown className="header-dropdown">
                {/*<PopupMenuToggle bsRole="toggle" className='menu-button' >Z</PopupMenuToggle>*/}
                <PopupMenuToggle bsRole="toggle">
                    <Glyphicon className='menu-button' glyph="menu-hamburger"/>
                    {/*<span className='menu-button'>U</span>*/}
                </PopupMenuToggle>
                <Dropdown.Menu bsRole="menu" className="popup-menu">
                    {/*<MenuItem eventKey="1" onClick={this.toggleProperties}>*/}
                    {/*<Glyphicon className='menu-item-icon' glyph="cog" /> Properties*/}
                    {/*</MenuItem>*/}
                    {/*<MenuItem divider />*/}
                    <MenuItem eventKey="2" onClick={this.close}>
                        <span className='menu-item-close'>×</span> Close
                    </MenuItem>
                </Dropdown.Menu>
            </Dropdown>
        );

        return (
            /* jshint ignore: start */
            <div className={classes} style={styles} onMouseDown={e => this.handleMove(e, false)}
                 ref={el => this.ref = el}
            >
                <header onMouseDown={e => this.handleMove(e, true)}>
                    {menu}
                    <div className='close-button' onClick={this.close}>×</div>
                    {/*{propertiesContent != null ? (*/}
                    {/*<OverlayTrigger trigger="click" placement="top" overlay={propertiesContent}>*/}
                    {/*<Glyphicon className='properties-button' glyph="cog" />*/}
                    {/*</OverlayTrigger>*/}
                    {/*) : null}*/}
                    <OverlayTrigger show={true} placement="top"
                                    overlay={<Tooltip placement="top" positionLeft={0}>{this.window.title}</Tooltip>}>
                        <div className='title'>{this.window.title}</div>
                    </OverlayTrigger>
                </header>
                <div className='content' onMouseDown={this.handlePropagation} ref='content'>
                    {content}
                </div>
                <div className='resize s-resize' onMouseDown={this.handleResize}/>
                <div className='resize e-resize' onMouseDown={this.handleResize}/>
                <div className='resize w-resize' onMouseDown={this.handleResize}/>
                <div className='resize sw-resize' onMouseDown={this.handleResize}/>
                <div className='resize se-resize' onMouseDown={this.handleResize}/>
            </div>
            /* jshint ignore: end */
        );
    }

}
