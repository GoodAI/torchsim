import * as THREE from 'three';
import { ClusterObserverData } from '../../ClusterObserver';
import { DataProvider } from '../Helpers/DataProvider';

export interface SceneSubject {
    readonly sceneObject: THREE.Object3D;
    update(clusterPositions: THREE.Vector3[], dataProvider: DataProvider);
}