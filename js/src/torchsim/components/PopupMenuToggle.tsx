import * as React from "react";
import * as ReactDOM from 'react-dom';


interface PopupMenuToggleProps {
    bsRole: string;
    className?: string;
    onClick?: () => void;
}

interface PopupMenuToggleState {
}

export class PopupMenuToggle extends React.Component<PopupMenuToggleProps, {}> {
    state: PopupMenuToggleState;

    constructor(props: PopupMenuToggleProps) {
        super(props);
        this.state = {
        };
    }

    render() {
        return (
            <span onClick={() => this.props.onClick()}>
            {this.props.children}
            </span>
        );
    }
}
