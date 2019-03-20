'use strict';
import * as React from "react";
import * as ReactDOM from "react-dom";

import {App} from "./torchsim/App";

import '../css/wm.scss'
import '../css/styles.scss'

const wrapper = document.getElementById("app");
wrapper ? ReactDOM.render(<App />, wrapper) : false;

console.log("UI App started");
