import * as React from "react";
import {MyRef} from "./TextBox";

interface CheckProps {
    value: boolean;
    submitHandler: (string) => void
    readOnly?: boolean
    disabled?: boolean
    className?: string;
}

interface CheckState {
}

export class CheckBox extends React.Component<CheckProps, CheckState> {
    private input: MyRef;
    state: CheckState;

    constructor(props: CheckProps) {
        super(props);
        this.input = new MyRef();
        this.state = {};
    }

    handleChange = (event) => {
        event.preventDefault();
        event.stopPropagation();

        this.props.submitHandler(event.target.checked);
    };

    render() {
        return (
            <input className={this.props.className} type="checkbox" ref={this.input.setRef}
                   checked={this.props.value} onChange={this.handleChange} disabled={this.props.disabled}
            />
        );
    }
}
