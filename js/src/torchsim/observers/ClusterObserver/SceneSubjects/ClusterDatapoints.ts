import * as THREE from 'three';
import { DataProvider } from '../Helpers/DataProvider';
import { InvalidArgumentException } from '../../../exceptions';
import { ResizableBufferGeometry } from '../Helpers/ResizableBufferGeometry';
import { SceneSubject } from './SceneSubject';

export interface ClusterDatapointsData {
    cluster_datapoints?: number[][];
    cluster_datapoints_cluster_ids?: number[];
}

export class ClusterDatapoints implements SceneSubject {
    private nClusterCenters: number;
    private points: ResizablePoints;
    private lines: ResizableLineSegments;
    private shapeGroup: THREE.Group;

    constructor(nClusterCenters: number) {
        this.nClusterCenters = nClusterCenters;
        this.points = ClusterDatapoints.buildPoints();
        this.lines = ClusterDatapoints.buildLines();
        this.shapeGroup = ClusterDatapoints.buildShapeGroup(this.points, this.lines);
    }

    public get sceneObject(): THREE.Object3D {
        return this.shapeGroup;
    }

    public update = (clusterPositions: THREE.Vector3[], dataProvider: DataProvider) => {
        const data = dataProvider.coData.pca.cluster_datapoints;

        if (data == undefined)
            throw new InvalidArgumentException(`${ClusterDatapoints.name}.update called without data being set.`);

        if (data.cluster_datapoints && data.cluster_datapoints_cluster_ids) {
            const cdcIds = data.cluster_datapoints_cluster_ids;
            const ccPositions = clusterPositions;
            const cdPositions = data.cluster_datapoints.map(arr => new THREE.Vector3(arr[0], arr[1], arr[2]));
            const ccColors = DataProvider.generateColors(this.nClusterCenters);
            const cdColors = Array.from(cdcIds, id => ccColors[id]);

            this.points.update(cdPositions, cdColors);
            this.lines.update(cdcIds, ccPositions, ccColors, cdPositions, cdColors);
        }
    };

    private static buildPoints(): ResizablePoints {
        const params: THREE.PointsMaterialParameters = {
            // map: new THREE.TextureLoader().load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/sprites/disc.png'),
            // alphaTest: 0.5,
            // transparent: true,
            size: 1,
            sizeAttenuation: true,
        };
        return new ResizablePoints(params);
    }

    private static buildLines(): ResizableLineSegments {
        const params: THREE.LineBasicMaterialParameters = {
            linewidth: 1, // larger values have no effect :(
        };
        return new ResizableLineSegments(params);
    }

    private static buildShapeGroup(points: ResizablePoints, lines: ResizableLineSegments): THREE.Group {
        const group = new THREE.Group();
        group.add(points);
        group.add(lines);
        return group;
    }

}

class ResizablePoints extends THREE.Points {
    public geometry: ResizableBufferGeometry;
    public material: THREE.PointsMaterial;

    constructor(params?: THREE.PointsMaterialParameters) {
        super(new ResizableBufferGeometry(), new THREE.PointsMaterial({...params, vertexColors: THREE.VertexColors}));
    }

    public update = (positions: THREE.Vector3[], colors: THREE.Color[]) => {
        if (positions.length !== colors.length)
            throw new InvalidArgumentException("Position and color arrays don't have the same length.");

        this.geometry.copyVector3sToAttribute(positions, ResizableBufferGeometry.PositionAttribOptions);
        this.geometry.copyColorsToAttribute(colors);
        this.geometry.computeBoundingBox();
    }
}

class ResizableLineSegments extends THREE.LineSegments {
    public geometry: ResizableBufferGeometry;
    public material: THREE.LineBasicMaterial;
    private nPoints: number;

    constructor(params?: THREE.LineBasicMaterialParameters) {
        super(new ResizableBufferGeometry(), new THREE.LineBasicMaterial({
            ...params,
            vertexColors: THREE.VertexColors
        }));
        this.nPoints = -1;
    }

    public update = (clusterDataIds: number[], clusterCenterPositions: THREE.Vector3[], clusterCenterColors: THREE.Color[], clusterDataPositions: THREE.Vector3[], clusterDataColors: THREE.Color[]) => {
        if (clusterCenterPositions.length !== clusterCenterColors.length)
            throw new InvalidArgumentException("Position and color arrays don't have the same length.");
        if (clusterDataIds.length !== clusterDataPositions.length)
            throw new InvalidArgumentException("Id and position arrays don't have the same length.");

        const nClusterCenters = clusterCenterPositions.length;
        const nPoints = clusterCenterPositions.length + clusterDataPositions.length;

        if (this.nPoints != nPoints) {
            this.geometry.setIndexAttribute(ResizableLineSegments.buildIndexArray(clusterDataIds, nClusterCenters));
        }

        this.geometry.copyVector3sToAttribute(clusterCenterPositions.concat(clusterDataPositions), ResizableBufferGeometry.PositionAttribOptions);
        this.geometry.copyColorsToAttribute(clusterCenterColors.concat(clusterDataColors));
        this.geometry.computeBoundingBox();
    };

    private static buildIndexArray(clusterDataIds: number[], nClusterCenters: number): number[] {
        const res = [];
        for (let i = 0; i < clusterDataIds.length; i++) {
            res.push(clusterDataIds[i], i + nClusterCenters);
        }
        return res;
    }
}
