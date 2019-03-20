import * as THREE from 'three';
import { SceneSubject } from './SceneSubject';
import { InvalidArgumentException } from '../../../exceptions';
import { DataProvider } from '../Helpers/DataProvider';

export interface SpringLinesData {
    // cluster_similarities?: number[][];
}

export class SpringLines implements SceneSubject {
    private static readonly CylinderHeight: number = 1;
    private static readonly CylinderRadius: number = 0.1;
    private static readonly RadialSegments: number = 8;
    private static readonly ShapeColor: THREE.Color = new THREE.Color('black');
    private static readonly CylinderGeometry: THREE.CylinderBufferGeometry =
        new THREE.CylinderBufferGeometry(SpringLines.CylinderRadius, SpringLines.CylinderRadius, SpringLines.CylinderHeight, SpringLines.RadialSegments);
    private static readonly InitialDirection: THREE.Vector3 = new THREE.Vector3(0, 1, 0);
    private static readonly MinConnectionStrength: number = 0.05;

    private springLinesGroup: THREE.Group;

    constructor(nClusterCenters: number) {
        const nLines = (nClusterCenters * (nClusterCenters - 1)) / 2;
        this.springLinesGroup = SpringLines.buildShapeGroup(nLines);
    }

    public get sceneObject(): THREE.Object3D {
        return this.springLinesGroup;
    }

    private get springLines(): THREE.Mesh[] {
        return this.springLinesGroup.children as THREE.Mesh[];
    }

    public update = (clusterPositions: THREE.Vector3[], dataProvider: DataProvider) => {
        const data = dataProvider.coData.spring_lines;
        const clusterSimilarities = dataProvider.clusterSimilarities;

        if (data == undefined)
            throw new InvalidArgumentException(`${SpringLines.name}.update called without data being set.`);

        if (clusterSimilarities == undefined) {
            this.sceneObject.visible = false;
            return;
        }

        this.sceneObject.visible = true;

        const cylinders = this.springLines;
        const quaternion = new THREE.Quaternion();
        const direction = new THREE.Vector3();

        let cIdx = 0;
        for (let i = 0; i < clusterPositions.length; i++) {
            for (let j = i + 1; j < clusterPositions.length; j++) {
                const cylinder = cylinders[cIdx++];
                const similarity = Math.max(clusterSimilarities[i][j], clusterSimilarities[j][i]);
                const posA = clusterPositions[i];
                const posB = clusterPositions[j];

                cylinder.visible = !posA.equals(posB) && similarity >= SpringLines.MinConnectionStrength;

                if (!cylinder.visible)
                    continue;

                direction.copy(posA).sub(posB);
                const length = direction.length();

                direction.normalize();
                quaternion.setFromUnitVectors(SpringLines.InitialDirection, direction);
                cylinder.position.copy(posA).add(posB).multiplyScalar(0.5);
                cylinder.rotation.setFromQuaternion(quaternion);
                cylinder.scale.set(similarity, length, similarity);
            }
        }
    };

    private static buildSpringLineShape(): THREE.Mesh {
        const materialParams: THREE.MeshBasicMaterialParameters = { color: SpringLines.ShapeColor };
        const material = new THREE.MeshBasicMaterial(materialParams)
        const cylinderMesh = new THREE.Mesh(SpringLines.CylinderGeometry, material);
        return cylinderMesh;
    }

    private static buildShapeGroup(count: number): THREE.Group {
        const group = new THREE.Group();

        for (let i = 0; i < count; i++) {
            const shape = SpringLines.buildSpringLineShape();
            group.add(shape);
        }

        return group;
    }
}