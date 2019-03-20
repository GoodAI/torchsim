import * as React from 'react';
import { ObserverProps, ObserverUtils } from './Observer';
import { SceneManager } from './ClusterObserver/SceneManager';
import { PcaData } from './ClusterObserver/PositionCalculators/PcaProjector';
import { ClusterCentersData } from './ClusterObserver/SceneSubjects/ClusterCenters';
import { ForceDirectedSimulatorData } from './ClusterObserver/PositionCalculators/ForceDirectedSimulator';
import { SpringLinesData } from './ClusterObserver/SceneSubjects/SpringLines';
import { SequencesData } from './ClusterObserver/Helpers/DataProvider';
import { SplineArrowsData } from './ClusterObserver/SceneSubjects/SplineArrows';

export interface ClusterObserverData {
    n_cluster_centers: number;
    n_dims: number;
    n_sequences: number;
    sequence_length: number;
    projection_type: string;

    cluster_centers?: ClusterCentersData;
    fdsim?: ForceDirectedSimulatorData;
    pca?: PcaData;
    spring_lines?: SpringLinesData;
    spline_arrows?: SplineArrowsData;

    sequences: SequencesData;

    width: number;
    height: number;
}

export interface ClusterObserverProps extends ObserverProps {
    data: ClusterObserverData;
    properties: any;
}

interface ClusterObserverState {
}

export class ClusterObserver extends React.Component<ClusterObserverProps> {
    private canvasElement: HTMLCanvasElement;
    private sceneManager: SceneManager;
    private observerUtils = new ObserverUtils(this);

    state: ClusterObserverState = {};

    constructor(props: ClusterObserverProps) {
        super(props);
    }

    public componentDidMount() {
        // console.log('RECEIVED DATA', this.props.data);
        this.sceneManager = new SceneManager(this.canvasElement, this.props.data);
    }

    public componentDidUpdate(prevProps: ClusterObserverProps, prevState: ClusterObserverState) {
        // const shallowCopy = { ...this.props.data };
        this.sceneManager.update(this.props.data);
        this.sceneManager.render();
    }

    setContentRef = (element: HTMLElement) => {
        this.observerUtils.setContentRef(element);
        this.canvasElement = element as HTMLCanvasElement;
    };

    public render() {
        this.observerUtils.onRender();
        let content = this.props.data;
        return (<canvas
            className='cluster-observer-canvas'
            width={content.width + "px"}
            height={content.height + "px"}
            ref={this.setContentRef}
        />);
    }
}
