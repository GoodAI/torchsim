import * as React from "react";
import {ObserverProps, ObserverUtils} from "./Observer";

export interface ImageObserverProps extends ObserverProps {
    data: {
        src: string
    }
}

export class ImageObserver extends React.Component<ImageObserverProps, {}> {
    observerUtils = new ObserverUtils(this);

    constructor(props: ImageObserverProps) {
        super(props);
    }

    render() {
        this.observerUtils.onRender();
        return (
            <img src={this.props.data.src}
                 ref={(element) => this.observerUtils.setContentRef(element)}
            />
        );
    }
}
