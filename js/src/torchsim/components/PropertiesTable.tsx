import * as React from "react";
import {TextBox} from "./TextBox";
import {AppApi} from "../App";
import * as classNames from 'classnames';
import * as Glyphicon from 'react-bootstrap/lib/Glyphicon';
import {OptionalTextBox} from "./OptionalTextBox";
import * as Tooltip from 'react-bootstrap/lib/Tooltip';
import * as OverlayTrigger from 'react-bootstrap/lib/OverlayTrigger';

export enum PropertiesTablePropertyItemState {
    Enabled = "enabled",
    Disabled = "disabled",
    ReadOnly = "readonly"
}

export enum PropertiesTablePropertyItemType {
    Text = "text",
    Number = "number",
    Button = "button",
    Checkbox = "checkbox",
    Select = "select",
    CollapsibleHeader = "collapsible_header"
}

export enum PropertiesTablePropertyItemSourceType {
    Observer = "observer",
    Model = "model",
    Control = "control",
}

export interface PropertiesTablePropertyItem {
    id: number;
    type: PropertiesTablePropertyItemType;
    name: string;
    value: any;
    state: PropertiesTablePropertyItemState;
    values?: string[];
    optional: boolean;
    sourceType: PropertiesTablePropertyItemSourceType;
    hint: string;
}

export interface PropertiesTableProps {
    data: PropertiesTablePropertyItem[];
    appApi: AppApi;
}

interface PropertiesTableRow {
    item: PropertiesTablePropertyItem
    nested: boolean
}

export class PropertiesTable extends React.Component<PropertiesTableProps, {}> {
    constructor(props: PropertiesTableProps) {
        super(props);

    }

    updateValue = (propId, value) => {
        // this.props.onFocus(this.props.id, () => {
        this.props.appApi.sendPaneMessage(
            {
                event_type: 'property_updated',
                property_id: propId,
                value: value
            }
        );
    };

    renderPropertyValue = (prop: PropertiesTablePropertyItem) => {
        let updateWithReadOnlyCheck = (event, value) => {
            if (prop.state == PropertiesTablePropertyItemState.ReadOnly) {
                event.preventDefault();
                return;
            }
            this.updateValue(prop.id, value)
        }

        const MaybeOptionalTextBox = prop.optional ? OptionalTextBox : TextBox;
        switch (prop.type) {
            case PropertiesTablePropertyItemType.Text:
                return <MaybeOptionalTextBox
                    disabled={prop.state == PropertiesTablePropertyItemState.Disabled}
                    readOnly={prop.state == PropertiesTablePropertyItemState.ReadOnly}
                    value={prop.value}
                    submitHandler={(value) => this.updateValue(prop.id, value)}
                />;
            case PropertiesTablePropertyItemType.Number:
                return <MaybeOptionalTextBox
                    disabled={prop.state == PropertiesTablePropertyItemState.Disabled}
                    readOnly={prop.state == PropertiesTablePropertyItemState.ReadOnly}
                    value={prop.value}
                    submitHandler={(value) => this.updateValue(prop.id, value)}
                    validateHandler={(value) => value.match(/^-?[0-9]*([.][0-9]*)?$/i)}
                />;
            case PropertiesTablePropertyItemType.Button:
                return <button
                    disabled={prop.state != PropertiesTablePropertyItemState.Enabled}
                    className="btn btn-xs btn-primary"
                    onClick={() => this.updateValue(prop.id, "clicked")}
                >{prop.value}</button>
            case PropertiesTablePropertyItemType.Checkbox:
                return <label className="checkbox checkbox-inline">
                    <input
                        type="checkbox"
                        checked={prop.value}
                        onChange={(event) => updateWithReadOnlyCheck(event, !prop.value)}
                        disabled={prop.state == PropertiesTablePropertyItemState.Disabled}
                    />
                    &nbsp;
                </label>;
            case PropertiesTablePropertyItemType.Select:
                return <select className="form-control"
                               onChange={(event) => updateWithReadOnlyCheck(event, event.target.value)}
                               value={prop.value}
                               disabled={prop.state == PropertiesTablePropertyItemState.Disabled || prop.state == PropertiesTablePropertyItemState.ReadOnly}
                >
                    {prop.values.map((name, id) => <option value={id}>{name}</option>)}
                </select>;
        }
    };

    componentDidUpdate(prevProps: Readonly<PropertiesTableProps>, prevState: Readonly<{}>, snapshot?: any): void {

    }

    private preparePropertiesItems(data: PropertiesTablePropertyItem[]) {
        let result: PropertiesTableRow[] = [];
        let expanded = true;
        let nested = false;
        let pushItem = item => {
            result.push({
                item: item,
                nested: nested
            });
        };

        for (let item of data) {
            if (item.type === PropertiesTablePropertyItemType.CollapsibleHeader) {
                nested = true;
                expanded = item.value;
                pushItem(item);
            } else {
                if (expanded) {
                    pushItem(item);
                }
            }
        }
        return result;
    }

    render() {
        const propItems = this.preparePropertiesItems(this.props.data);


        const renderRow = (row: PropertiesTableRow) => {
            const prop = row.item;
            const hint = (
                <div className="table-properties-hint">
                    <span dangerouslySetInnerHTML={{__html:prop.hint}}/>
                    <div className="source-type">
                        <span className="name">Source type:</span>
                        <span className="value">{prop.sourceType}</span>
                    </div>
                </div>);
            if (prop.type == PropertiesTablePropertyItemType.CollapsibleHeader) {
                return ([
                    <td className="table-properties-header" colSpan={2}
                        onClick={(event) => this.updateValue(prop.id, !prop.value)}>
                        <Glyphicon className="icon" glyph={prop.value ? "triangle-bottom" : "triangle-right"}/>
                        {/*<Glyphicon className="icon" glyph={prop.value ? "collapse-down" : "expand"} />*/}
                        {prop.name}
                    </td>,
                ])

            } else {
                return ([
                    <td className={classNames("table-properties-name", {'nested': row.nested})}>

                        <OverlayTrigger placement="left"
                                        overlay={<Tooltip placement="left" positionLeft={0}>{hint}</Tooltip>}>
                            <span>
                                {prop.sourceType == PropertiesTablePropertyItemSourceType.Model && (
                                    <Glyphicon className="source-type-icon" glyph="tag"/>)}
                                {prop.name}
                                </span>
                        </OverlayTrigger>

                    </td>,
                    <td className="table-properties-value">{this.renderPropertyValue(prop)}</td>
                ])
            }
        };

        return (
            <table className="table table-condensed table-properties table-hover">
                <tbody>
                {propItems.map(row =>
                    <tr key={row.item.id}>
                        {renderRow(row)}
                    </tr>
                )}
                </tbody>
            </table>
        )
    }

}
