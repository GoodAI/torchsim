'use strict';

import {WindowManagerModel} from "../models/WindowManagerModel";

var _ = require('lodash');
var $ = require('jquery');
import * as React from "react";
import * as ReactDOM from "react-dom";
import * as PropTypes from 'prop-types';
// var CSSTransitionGroup = require('react/addons').addons.CSSTransitionGroup;

// var Window = require('./window');
import {Window} from './Window'
import {WindowModel} from "../models/WindowModel";
import {SignalConnection} from "typed-signals";
import {BatchedExecution} from "../GuiUtils";
import {DEFAULT_UI_BATCH_DELAY} from "../../torchsim/App";


export interface Offset {
    top: number,
    left: number
}

export interface DesktopProps {
    manager: any;
}

interface DesktopState {
    offset: Offset
}


export class Desktop extends React.Component<DesktopProps, {}> {
    private manager: WindowManagerModel;
    private batchedForceUpdate: BatchedExecution =
        new BatchedExecution(() => this.forceUpdate(), DEFAULT_UI_BATCH_DELAY);

    statics = {
        Manager: WindowManagerModel,
        Window: WindowModel
    };
    private managerChangeSignal: SignalConnection;

    componentDidMount() {
        this.manager = this.props.manager;
        this.managerChangeSignal = this.manager.signals.change.connect(() => this.batchedForceUpdate.run());

        var el = $(ReactDOM.findDOMNode(this));
        this.setState({offset: el.offset()});
    }

    componentWillUnmount() {
        this.managerChangeSignal.disconnect();
        // this.manager.off('change', this.forceUpdate);
    }

    state = {
        offset: {
            top: 0,
            left: 0
        }
    }

    render() {
        let windows = this.props.manager.openWindows().map((window) => (
                <Window key={window.id} offset={this.state.offset} window={window}/>
            )
        );
        return (
            <div className='window-manager desktop'>
                <div className='windows'>{windows}</div>
            </div>
        );
    }

}
