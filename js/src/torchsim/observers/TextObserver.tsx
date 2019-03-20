import * as React from "react";

export interface TextObserverProps {
    data: string;
}


export class TextObserver extends React.Component<TextObserverProps, {}> {
    constructor(props: TextObserverProps) {
        super(props);
    }

    render() {
        const content = this.props.data;
        return (
            <div dangerouslySetInnerHTML={{__html: content}}/>
        );
    }
}
