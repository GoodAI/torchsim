import * as Tooltip from 'react-bootstrap/lib/Tooltip';
import * as OverlayTrigger from 'react-bootstrap/lib/OverlayTrigger';
import * as classNames from 'classnames';
import * as React from 'react';
import {ObserverProps, ObserverUtils} from "./Observer";
import {formatRawValue, prepareTooltip} from './MemoryBlockObserver';

import * as _ from 'lodash'

interface Params {
    scale: number,
    projection: {
        width: number,
        height: number,
        items_per_row: number
    }
}

export interface HierarchicalObserverGroupImageData {
    src: string
    values: {
        data: number[][],
        width: number,
        height: number
    },
}

export interface HierarchicalObserverGroupData {
    images: HierarchicalObserverGroupImageData[]
    params: Params
}

enum GroupsStacking {
    VERTICAL = "vertical",
    HORIZONTAL = "horizontal"
}

export interface HierarchicalObserverData {
    groups_stacking: GroupsStacking
    items_per_row: number
    groups: HierarchicalObserverGroupData[]
    caption: string
    properties: any
}

export interface HierarchicalObserverProps extends ObserverProps {
    data: HierarchicalObserverData
}

export class HierarchicalObserver extends React.Component<HierarchicalObserverProps, {}> {
    state = {
        hoverValue: ''
    };

    observerUtils = new ObserverUtils(this);

    constructor(props) {
        super(props);
    }

    onEvent = (event) => {
        switch (event.type) {
            case 'keydown':
            case 'keypress':
                event.preventDefault();
                break;
            case 'keyup':
                this.props.appApi.sendPaneMessage(
                    {
                        event_type: 'KeyPress',
                        key: event.key,
                        key_code: event.keyCode,
                    }
                );
                break;
        }
    };

    mouseMove = (group_idx, image_idx, e) => {
        // Update hover text
        let data = this.props.data.groups[group_idx].images[image_idx].values.data;
        let params = this.props.data.groups[group_idx].params;

        let [xRaw, yRaw, width, height, px, py, position] = prepareTooltip(data, params, e)
        let [x, y, value] = [xRaw, yRaw, '????'];

        this.props.appApi.sendPaneRequest<any>('GetHierarchicalValue',
            {
                x: xRaw,
                y: yRaw,
                group_idx: group_idx,
                image_idx: image_idx
            }
        ).then(
            (data) => {
                // console.log("GetMemoryBlockValue response", data);
                value = data.value;
                this.setState({
                    'hoverValue': formatRawValue(x, y, width, height, value, px, py, position, this.props.name)
                });
            },
            (err) => {
                console.error("GetMemoryBlockValue error", err);
            }
        );


        this.setState({
            'hoverValue': formatRawValue(x, y, width, height, value, px, py, position, this.props.name)
        });
    };

    render() {
        // console.log("Hierarchical render");
        this.observerUtils.onRender();
        let content = this.props.data;
        let tooltip = clusterId => (
            <Tooltip placement="top" className="memory-block-tooltip">
                {this.state.hoverValue}
                <div className="text-left">cluster: {clusterId}</div>
            </Tooltip>

        );

        // );
        // TODO: Width and height dimensions are swapped

        // Rearrange into rows of columns (we'll create a table next).

        if (content.groups.length == 0)
            return <div></div>;

        let clusters = content.groups[0].images.length;

        const render_memory_block = (clusterId: number, groupId: number) => {
            let group = content.groups[groupId];
            let image = group.images[clusterId];
            const height = image.values.height * group.params.scale;
            const width = image.values.width * group.params.scale;
            const className = classNames("group", {
                    'horizontal': this.props.data.groups_stacking == GroupsStacking.HORIZONTAL,
                    'vertical': this.props.data.groups_stacking == GroupsStacking.VERTICAL
                }
            );
            return <div key={groupId + "." + clusterId} className={className}>
                <OverlayTrigger placement="top" overlay={tooltip(clusterId)}>
                    <img
                        className="memory-block-image"
                        src={image.src}
                        width={width + "px"}
                        height={height + "px"}
                        onMouseMove={(e) => this.mouseMove(groupId, clusterId, e)}
                    />
                </OverlayTrigger>
            </div>
        };

        const render_cluster = (clusterId: number) => {
            if (clusterId >= clusters) {
                return <div></div>;
            }
            let groups = content.groups.length;
            return <div key={clusterId} className="cluster">
                {_.range(groups).map(groupId => render_memory_block(clusterId, groupId))}
            </div>
        };

        const columns = this.props.data.items_per_row;
        const rows = Math.ceil(clusters / columns);

        return (
            <div className="hierarchical-observer">
                <table ref={(element) => this.observerUtils.setContentRef(element)}>
                    <tbody>
                    {_.range(rows).map(row => {
                        return <tr key={row}>
                            {_.range(columns).map(column => {
                                return <td>
                                    {render_cluster(row * columns + column)}
                                </td>
                            })}
                        </tr>
                    })}

                    </tbody>
                </table>
            </div>
        );

    }
}
