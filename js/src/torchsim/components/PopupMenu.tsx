import * as React from "react";
import * as classNames from 'classnames';


interface PopupMenuProps {
    bsRole: string;
    className?: string;
}

interface PopupMenuState {
}

export class PopupMenu extends React.Component<PopupMenuProps, {}> {
    state: PopupMenuState;

    constructor(props: PopupMenuProps) {
        super(props);
        this.state = {
        };
    }

    render() {
        let classes = classNames("dropdown-menu open", this.props.className);
        return (
            <ul className={classes}>
                {this.props.children}
            </ul>
        );
    }
}
