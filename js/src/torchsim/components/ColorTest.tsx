import * as React from "react";
import { TextBox } from "./TextBox";
import { DataProvider } from "../observers/ClusterObserver/Helpers/DataProvider";

export class ColorTest extends React.Component<any, any> {
    state = {
        number: 100
    };

    render() {
        let size = 20;
        let colors = DataProvider.generateColors(this.state.number).map(c => c.getHexString());
        return <div>
            <TextBox value={"" + this.state.number} submitHandler={(v) => this.setState({number: parseInt(v)})}/>
            <br/>
            {colors.map((c, i) =>
                <div style={{display: 'inline-block', width: `${size}px`, height: `${size}px`, backgroundColor: `#${c}`}}></div>
            )}
        </div>
    }
}
