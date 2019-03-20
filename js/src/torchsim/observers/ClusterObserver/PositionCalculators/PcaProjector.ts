import { Vector3 } from 'three';
import { InvalidArgumentException } from '../../../exceptions';
import { PositionCalculator } from './PositionCalculator';
import { ClusterDatapointsData } from '../SceneSubjects/ClusterDatapoints';
import { DataProvider } from '../Helpers/DataProvider';

export interface PcaData {
    cluster_centers?: number[][];
    cluster_datapoints?: ClusterDatapointsData;
}

export class PcaProjector implements PositionCalculator {

    private clusterPositions: Vector3[];

    constructor(nPoints: number) {
        this.clusterPositions = Array.from({length: nPoints}, () => new Vector3());
    }

    public update = (dataProvider: DataProvider): Vector3[] => {
        const data = dataProvider.coData.pca;

        if (data == undefined)
            throw new InvalidArgumentException(`${PcaProjector.name}.update called without data being set.`);
        if (data.cluster_centers != undefined) {
            this.clusterPositions.forEach((vect: Vector3, i: number) => vect.fromArray(data.cluster_centers[i]));
        }
        return this.clusterPositions;
    }
}

// import { InvalidStateException } from '../../../exceptions';
// import { Matrix} from 'ml-matrix';
// import { AsyncProjectionCalculationInput, AsyncProjectionCalculationResult } from './PcaWorker'
// import PcaWorker = require('worker-loader!./PcaWorker');

// Not used anymore - PCA is computed on backend
// export class PCA {
//     public readonly nDims: number = 3;
//     private pcaCalculated: boolean = false;
//     private means: number[];
//     private stdDevs: number[];
//     private U: Matrix;
//     private pcaWorker: PcaWorker;
//     private workerIsRunning: boolean;

//     constructor() {
//         this.pcaWorker = this.buildPcaCalculationWorker();
//         this.workerIsRunning = false;
//     }

//     public get isReady(): boolean {
//         return this.pcaCalculated;
//     }

//     public calculateProjection = (dataset: Matrix) => {
//         const input: AsyncProjectionCalculationInput = {
//             dataset: dataset,
//             maxDims: this.nDims,
//         }

//         if (this.workerIsRunning) {
//             this.pcaWorker.terminate();
//             this.pcaWorker = this.buildPcaCalculationWorker();
//             console.log('Terminated ongoing PCA calculation');
//         }

//         this.workerIsRunning = true;
//         this.pcaWorker.postMessage(input);
//         console.log(`Started a new background PCA calculation from (${dataset.rows}x${dataset.columns}) dataset`);
//     }

//     public normalizeInplace = (data: Matrix): Matrix => {
//         if (!this.pcaCalculated)
//             throw new InvalidStateException('Called normalizeInplace before computing PCA projection matrix');
//         data.subRowVector(this.means);
//         data.divRowVector(this.stdDevs);
//         return data;
//     }

//     public project = (normalizedData: Matrix) => {
//         if (!this.pcaCalculated)
//             throw new InvalidStateException('Called project before computing PCA projection matrix');
//         return normalizedData.mmul(this.U);
//     }

//     private buildPcaCalculationWorker = (): PcaWorker => {
//         const worker = new PcaWorker();
//         worker.onmessage = (ev: MessageEvent) => {
//             const response: AsyncProjectionCalculationResult = ev.data;

//             this.means = response.means;
//             this.stdDevs = response.stdDevs;
//             this.U = response.U;
//             this.workerIsRunning = false;
//             this.pcaCalculated = true;

//             console.log('PCA calculation done, explained variance:');
//             console.log(`3D: ${response.threeDimExplainedVar.toFixed(2)}%, 2D: ${response.twoDimExplainedVar.toFixed(2)} %`);
//         };

//         return worker;
//     }
// }