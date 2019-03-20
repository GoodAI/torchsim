import * as THREE from 'three';
import * as _ from "lodash";
import { ClusterObserverData } from "../../ClusterObserver";
import { InvalidStateException, InvalidArgumentException } from "../../../exceptions";

type clusterSimType = number[][];
type seqsType = number[][];
type seqsOccurencesType = number[];

class Lazy<T> {
    private val: T;
    private initFn: () => T;

    constructor(initFn: () => T) {
        this.initFn = initFn;
    }

    public get value(): T {
        return this.val || (this.val = this.initFn());
    }
}

export interface SequencesData {
    sequences?: seqsType;
    occurrences?: seqsOccurencesType;
}

export class DataProvider {
    private static readonly DefaultLazy: Lazy<any> = new Lazy<any>(() => undefined);

    private _coData?: ClusterObserverData;
    private similarities?: Lazy<clusterSimType> = DataProvider.DefaultLazy;

    public get coData(): ClusterObserverData {
        DataProvider.throwIfUndefined(this._coData);
        return this._coData;
    }

    public get clusterSimilarities(): clusterSimType {
        return this.similarities.value;
    }

    public update = (coData: ClusterObserverData) => {
        this._coData = coData;
        this.similarities = DataProvider.createLazySimilarities(coData) || this.similarities;
    }

    public static generateColors(count: number): THREE.Color[] {
        const saturation = 1;
        const maxHueSteps = Math.min(count, 25);
        const lMax = 0.75;
        const lMin = 0.25;
        const center = 0.5;
        const bankCount = Math.ceil(count / maxHueSteps);
        const ls = [center];
        const lsCount = Math.ceil((bankCount - 1) / 2);
        for (let i = 1; i <= lsCount; i++) {
            ls.push(center - (center - lMin) * i / lsCount);
            ls.push(center + (lMax - center) * i / lsCount);
        }
        return _.range(count)
            .map(i => {
                const hue = (i % maxHueSteps) / maxHueSteps;
                const bank = Math.ceil((i + 1) / maxHueSteps) - 1;
                return new THREE.Color().setHSL(hue, saturation, ls[bank])
            });
    }

    private static createLazySimilarities(coData: ClusterObserverData): Lazy<clusterSimType> {
        const seqsData = coData.sequences;
        if (seqsData.sequences == undefined || seqsData.occurrences == undefined)
            return undefined;

        return new Lazy(() => {
            return computeClusterSimilarities(coData.n_cluster_centers, coData.sequences);
        })
    }

    private static throwIfUndefined(arg: any) {
        if (arg == undefined) {
            throw new InvalidStateException(`Call update on ${DataProvider.name} before getting values from it.`);
        }
    }
}

function createArray<T>(length: number, initFn: () => T): T[] {
    return Array.from({length: length}, initFn);
}

function computeClusterSimilarities(nClusterCenters: number, data: SequencesData): clusterSimType {
    const res = createArray(nClusterCenters, () => createArray(nClusterCenters, () => 0));
    const sequences = data.sequences;
    const occurrences = data.occurrences;

    if (sequences.length != occurrences.length)
        throw new InvalidArgumentException(`Different number of sequences vs sequences frequencies (${sequences.length} vs ${occurrences.length})`);

    for (let i = 0; i < sequences.length; ++i) {
        const freq = occurrences[i];

        if (freq == 0)
            continue;

        const seq = sequences[i];

        let prev = seq[0];
        for (let j = 1; j < seq.length; ++j) {
            let curr = seq[j];
            res[prev][curr] += freq;
            prev = curr;
        }
    }

    let max = 0;
    for (let i = 0; i < nClusterCenters; ++i) {
        for (let j = 0; j < nClusterCenters; ++j) {
            const val = res[i][j];
            max = Math.max(max, val);
        }
    }

    if (max > 0) {
        for (let i = 0; i < nClusterCenters; ++i) {
            for (let j = 0; j < nClusterCenters; ++j) {
                res[i][j] /= max;
            }
        }
    }

    return res;
}