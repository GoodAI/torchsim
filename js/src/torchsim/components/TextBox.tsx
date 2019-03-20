import * as React from "react";
import classNames from "classnames"

export class MyRef {
    private _ref;

    constructor() {
        this._ref = null;
    }

    getRef() {
        return this._ref;
    }

    setRef = (ref) => {
        this._ref = ref;
    }
}

interface TextProps {
    value: string;
    validateHandler?: (string) => void
    submitHandler: (string) => void
    readOnly?: boolean
    disabled?: boolean
    className?: string;
}

interface TextState {
    propsValue: string;
    actualValue: string;
    isEdited: boolean;
}

export class TextBox extends React.Component<TextProps, {}> {
    private textInput: MyRef;
    state: TextState;

    constructor(props: TextProps) {
        super(props);
        this.textInput = new MyRef();
        this.state = {
            propsValue: props.value,
            actualValue: props.value,
            isEdited: false
        };
    }

    handleChange = (event) => {
        if( this.props.readOnly ) {
            event.preventDefault();
            return;
        }
        let newValue = event.target.value;
        if (this.props.validateHandler && !this.props.validateHandler(newValue)) {
            event.preventDefault();
        } else {
            this.setState({actualValue: newValue});
        }
    };

    handleKeyPress = (event) => {
        if (event.key === "Enter") {
            let ref = this.textInput.getRef();
            if (ref) ref.blur(); // Blur invokes submit
        }
    };

    onBlur = () => {
        this.setState({isEdited: false}, () => {
            if (this.props.submitHandler) {
                this.props.submitHandler(this.state.actualValue);
            }
        });
    };

    onFocus = () => {
        this.setState({isEdited: true});
    };

    focus() {
        let ref = this.textInput.getRef();
        if (ref) ref.focus();
    }

    componentWillReceiveProps(nextProps) {
        if (this.state.propsValue !== nextProps.value || !this.state.isEdited) {
            let newState = this.state.isEdited
                ? {propsValue: nextProps.value}
                : {propsValue: nextProps.value, actualValue: nextProps.value};
            this.setState(newState);
        }
    }

    render() {
        return (
            <input className={classNames(this.props.className, "form-control", "input-sm")} type="text" ref={this.textInput.setRef}
                   value={this.state.actualValue == null ? '' : this.state.actualValue} onChange={this.handleChange}
                   onKeyPress={this.handleKeyPress} onBlur={this.onBlur} onFocus={this.onFocus}
                disabled={this.props.disabled}
            />
        );
    }
}
