import * as Tooltip from 'react-bootstrap/lib/Tooltip';
import * as OverlayTrigger from 'react-bootstrap/lib/OverlayTrigger';
import * as Overlay from 'react-bootstrap/lib/Overlay';
import * as React from 'react';
import {ObserverProps, ObserverUtils} from "./Observer";

interface Params {
    scale: number,
    projection: {
        width: number,
        height: number,
        items_per_row: number
    }
    current_ptr?: number[]
}

export interface MemoryBlockObserverData {
    properties: any;
    caption: string
    params: Params
    src: string,
    values: {
        data: number[][],
        width: number,
        height: number
    },
}

export interface MemoryBlockObserverProps extends ObserverProps {
    data: MemoryBlockObserverData
}

export const prepareTooltip = (data, params, e) => {
    let getInverseTiling = (x: number, y: number, params: Params) => {
        // x = x / params.scale;
        // y = y / params.scale;
        let column = Math.floor(x / params.projection.width);
        let row = Math.floor(y / params.projection.height);
        position = row * params.projection.items_per_row + column;
        px = x % params.projection.width;
        py = y % params.projection.height;
        return [px, py, position];
    };

    let [width, height] = [data[0].length, data.length];
    let xRaw = Math.floor(e.nativeEvent.offsetX / params.scale);
    let yRaw = Math.floor(e.nativeEvent.offsetY / params.scale);
    let [px, py, position] = getInverseTiling(xRaw, yRaw, params);

    return [xRaw, yRaw, width, height, px, py, position]
};

export const formatRawValue = (x, y, width, height, value, px, py, position, name) => {
    let linearIndex = y * width + x;
    let precision = 3;

    function formatValue(value) {
        if (typeof(value) === 'number') {
            return (Math.abs(value) < Math.pow(10, -precision))
                ? value.toExponential(precision)
                : value.toFixed(precision);
        } else {
            return value;
        }
    }

    let line1 = `[${x}, ${y}] (${linearIndex}): ${formatValue(value)}`;
    let line2 = `Before projection: [${px}, ${py}] (${position})`;
    return <div className="text-left">
        <div className="memory-block-tooltip-header">{name}</div>
        {line1}
        <br/>
        {line2}
    </div>;
};

class CanvasImageBarrier {
    private _canvasElement: HTMLCanvasElement = null;
    private _image: HTMLImageElement = null;
    private draw: (canvas: HTMLCanvasElement, image: HTMLImageElement) => void;

    constructor(draw: (canvas: HTMLCanvasElement, image: HTMLImageElement) => void) {
        this.draw = draw;
    }

    set canvasElement(value: HTMLCanvasElement) {
        this._canvasElement = value;
        this.tryDrawImage();
    }

    set imageUrl(url: string) {
        let image = new Image;
        image.onload = () => {
            this._image = image;
            this.tryDrawImage();
        };
        image.src = url;
    }

    private tryDrawImage() {
        if (!this._canvasElement || !this._image) {
            // Element is not ready yet
            return;
        }
        this.draw(this._canvasElement, this._image);
    }
}

export class MemoryBlockObserver extends React.Component<MemoryBlockObserverProps, {}> {
    state = {
        scale: 1.,
        hoverValue: ''
    };

    observerUtils = new ObserverUtils(this);
    canvasImageBarrier: CanvasImageBarrier;

    constructor(props) {
        super(props);
        this.canvasImageBarrier = new CanvasImageBarrier(this.renderImage);
        this.updateImageUrl();
    }

    updateImageUrl() {
        this.canvasImageBarrier.imageUrl = this.props.data.src;
    }

    onEvent = (event) => {
        // if (!this.props.isFocused) {
        //     return;
        // }

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

    // componentDidMount() {
    //     EventSystem.subscribe('global.event', this.onEvent)
    // }

    // componentWillMount() {
    //     EventSystem.unsubscribe('global.event', this.onEvent)
    // }

    mouseMove = (e) => {
        // Update hover text
        let data = this.props.data.values.data;
        let params = this.props.data.params;

        let [xRaw, yRaw, width, height, px, py, position] = prepareTooltip(data, params, e)
        let [x, y, value] = [xRaw, yRaw, '????'];

        this.props.appApi.sendPaneRequest<any>('GetMemoryBlockValue',
            {
                x: xRaw,
                y: yRaw
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

    setContentRef = (element: HTMLElement) => {
        this.observerUtils.setContentRef(element);
        this.canvasImageBarrier.canvasElement = element as HTMLCanvasElement;
    };


    componentDidUpdate(prevProps: Readonly<MemoryBlockObserverProps>, prevState: Readonly<{}>, snapshot?: any): void {
        this.updateImageUrl();
    }

    renderImage = (canvas: HTMLCanvasElement, image: HTMLImageElement): void => {
        const ctx = canvas.getContext("2d");
        const scale = this.props.data.params.scale;
        ctx.imageSmoothingEnabled = false;
        ctx.setTransform(scale, 0, 0, scale, 0, 0);
        ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, image.width, image.height);
        const highlightedTiles = this.props.data.params.current_ptr;
        if (highlightedTiles) {
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            for (let tile of highlightedTiles) {
                const p = this.props.data.params.projection;
                const row = Math.floor(tile / p.items_per_row);
                const column = tile % p.items_per_row;
                let [x, y] = [column * p.width, row * p.height];

                ctx.lineWidth = 2;
                const segmentSize = 6;
                // const colors = ["#ff8000", "#000000", "#ff007f", "#ffffff"];
                const colors = ["#ff007f", "#ffffff"];
                for( let i = 0; i < colors.length; i++) {
                    ctx.strokeStyle = colors[i];
                    ctx.lineDashOffset = i * segmentSize;
                    ctx.setLineDash([segmentSize, (colors.length - 1) * segmentSize]);
                    ctx.strokeRect(x * scale, y * scale, p.width * scale, p.height * scale);
                }
            }
        }
    };

    render() {
        // console.log("Memory block render");
        this.observerUtils.onRender();
        let content = this.props.data;
        let tooltip = (
            <Tooltip placement="top" className="memory-block-tooltip">
                {this.state.hoverValue}
            </Tooltip>

        );
        const height = content.values.height * content.params.scale;
        const width = content.values.width * content.params.scale;

        return (
            <div>
                <OverlayTrigger placement="top" overlay={tooltip}>
                    <canvas
                        ref={this.setContentRef}
                        className="memory-block-image"
                        width={width + "px"}
                        height={height + "px"}
                        onMouseMove={this.mouseMove}
                    />
                </OverlayTrigger>
            </div>
        );

    }
}
