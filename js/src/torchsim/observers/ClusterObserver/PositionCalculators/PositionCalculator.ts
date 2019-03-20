import { Vector3 } from 'three';
import { DataProvider } from '../Helpers/DataProvider';

export interface PositionCalculator {
    update(dataProvider: DataProvider): Vector3[];
}