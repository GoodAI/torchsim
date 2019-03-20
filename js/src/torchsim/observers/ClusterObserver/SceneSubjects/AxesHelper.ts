import * as THREE from 'three';
import { DataProvider } from '../Helpers/DataProvider';
import { SceneSubject } from './SceneSubject';

export class AxesHelper implements SceneSubject {
    private axesHelper: THREE.AxesHelper;

    constructor() {
        const size = 10;
        this.axesHelper = new THREE.AxesHelper(size);
    }

    public get sceneObject(): THREE.Object3D {
        return this.axesHelper;
    }

    public update = (clusterPositions: THREE.Vector3[], dataProvider: DataProvider) => {
    };
}