import * as React from "react";
import {ObserverProps, ObserverUtils} from "./Observer";
import {PropertiesTable, PropertiesTablePropertyItem} from "../components/PropertiesTable";


export interface PropertiesObserverProps extends ObserverProps {
    data: PropertiesTablePropertyItem[]
}

export class PropertiesObserver extends React.Component<PropertiesObserverProps, {}> {
    observerUtils = new ObserverUtils(this);

    constructor(props: PropertiesObserverProps) {
        super(props);

    }
    render() {
        this.observerUtils.onRender();
        return (
            <div className="content-properties"
                 ref={(element) => this.observerUtils.setContentRef(element)}
            >
                <PropertiesTable data={this.props.data} appApi={this.props.appApi}/>
            </div>
        )
    }
}
