import * as React from "react";
import {MyRef, TextBox} from "./TextBox";
import {CheckBox} from "./CheckBox";

interface OptionalInputProps {
    value: string;
    validateHandler?: (string) => void
    submitHandler: (string) => void
    readOnly?: boolean
    disabled?: boolean
    className?: string;
}

interface OptionalInputState {
    propsIsValid: boolean;
    actualIsValid: boolean;
    isEdited: boolean;
    propsValue: string;
}

export class OptionalTextBox extends React.Component<OptionalInputProps, {}> {
    private textInput: MyRef;
    state: OptionalInputState;

    constructor(props: OptionalInputProps) {
        super(props);
        this.textInput = new MyRef();
        this.state = {
            propsIsValid: OptionalTextBox.isValid(props.value),
            actualIsValid: OptionalTextBox.isValid(props.value),
            isEdited: false,
            propsValue: null,
        };
    }

    static isValid(value: string): boolean {
        return !(value == null || value === '');
    }

    componentWillReceiveProps(nextProps: OptionalInputProps) {
        const isValid = OptionalTextBox.isValid(nextProps.value);
        const nextState = {};

        if (isValid && !this.state.isEdited) {
            Object.assign(nextState, { propsValue: nextProps.value });
        }

        if (this.state.isEdited) {
            Object.assign(nextState, { propsIsValid: isValid });
        } else {
            Object.assign(nextState, { propsIsValid: isValid, actualIsValid: isValid });
        }

        this.setState(nextState);
    }

    textBoxSubmitHandler = value => {
        this.setState({isEdited: false}, () => {
            this.props.submitHandler(value);
        });
    };

    checkBoxSubmitHandler = value => {
        if(this.props.readOnly) {
            return;
        }
        if (value) {
            this.setState({actualIsValid: value, isEdited: true}, () => {
                const textbox = this.textInput.getRef();
                textbox.focus();
            });
        } else {
            this.setState({actualIsValid: value, isEdited: false}, () => {
                this.props.submitHandler(null);
            });
        }
    };

    render() {
        return <div className='optional-container'>
            <TextBox
                ref={this.textInput.setRef}
                className="text-input"
                disabled={this.state.actualIsValid ? this.props.disabled : true}
                readOnly={this.props.readOnly}
                value={this.state.actualIsValid ? this.state.propsValue : ""}
                submitHandler={this.textBoxSubmitHandler}
                validateHandler={this.props.validateHandler}
            />
            <CheckBox className="optional-switch" value={this.state.actualIsValid}
                      disabled={this.props.disabled}
                      readOnly={this.props.readOnly}
                      submitHandler={this.checkBoxSubmitHandler}/>
        </div>
    }
}
