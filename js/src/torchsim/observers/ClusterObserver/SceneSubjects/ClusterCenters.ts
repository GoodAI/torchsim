import * as THREE from 'three';
import { DataProvider } from '../Helpers/DataProvider';
import { InvalidArgumentException } from '../../../exceptions';
import { SceneSubject } from './SceneSubject';

export interface ClusterCentersData {
    current_id: number;
    current_size_coef: number;
    size_coef: number;
    projections?: string[];
}

export class ClusterCenters implements SceneSubject {
    private static readonly ShapeRadius: number = 0.5;
    private static readonly ShapeNicesess: number = 16;
    private static readonly ShapeColor: THREE.Color = new THREE.Color('#2980b9');
    private static readonly CurrentShapeColor: THREE.Color = new THREE.Color('#d0be3f');
    private static readonly SphereGeometry = new THREE.SphereGeometry(ClusterCenters.ShapeRadius, ClusterCenters.ShapeNicesess, ClusterCenters.ShapeNicesess);
    // private static readonly SphereGeometry = new THREE.BoxBufferGeometry(ClusterCenters.ShapeRadius, ClusterCenters.ShapeRadius, ClusterCenters.ShapeRadius);

    private nClusterCenters: number;
    private groupsGroup: THREE.Group;
    private spheresGroup: THREE.Group;
    private spritesGroup: THREE.Group;

    private textureLoader: THREE.TextureLoader = new THREE.TextureLoader();

    constructor(nClusterCenters: number) {
        this.nClusterCenters = nClusterCenters;

        this.spheresGroup = ClusterCenters.buildShapeGroup(nClusterCenters, ClusterCenters.buildClusterCenterSphere);
        this.spritesGroup = ClusterCenters.buildShapeGroup(nClusterCenters, ClusterCenters.buildClusterCenterSprite);
        // console.log(this.spritesGroup.children);

        this.groupsGroup = new THREE.Group();
        this.groupsGroup.add(this.spheresGroup);
        this.groupsGroup.add(this.spritesGroup);
    }

    private get spheres(): THREE.Mesh[] {
        return this.spheresGroup.children as THREE.Mesh[];
    }

    private get sprites(): THREE.Sprite[] {
        return this.spritesGroup.children as THREE.Sprite[];
    }

    public get sceneObject(): THREE.Object3D {
        return this.groupsGroup;
    }

    public update = (clusterPositions: THREE.Vector3[], dataProvider: DataProvider) => {
        const data = dataProvider.coData.cluster_centers;

        if (data == undefined)
            throw new InvalidArgumentException(`${ClusterCenters.name}.update called without data being set.`);

        const showProjections = data.projections != undefined;

        if (showProjections) {
            this.updateSprites(clusterPositions, data);
        } else {
            this.updateSpheres(clusterPositions, data);
        }

        this.spheresGroup.visible = !showProjections;
        this.spritesGroup.visible = showProjections;

    }

    private updateSpheres = (clusterPositions: THREE.Vector3[], data: ClusterCentersData) => {
        const sizeCoef = data.size_coef;

        for (let i = 0; i < this.nClusterCenters; i++) {
            const clusterCenter = this.spheres[i];
            const material = clusterCenter.material as THREE.MeshPhongMaterial;

            clusterCenter.position.copy(clusterPositions[i]);

            if (i == data.current_id) {
                material.color.set(ClusterCenters.CurrentShapeColor);
                const currentCoef = data.current_size_coef * sizeCoef;
                clusterCenter.scale.set(currentCoef, currentCoef, currentCoef);
            } else {
                material.color.set(ClusterCenters.ShapeColor);
                clusterCenter.scale.set(sizeCoef, sizeCoef, sizeCoef);
            }
        }
    }

    private updateSprites = (clusterPositions: THREE.Vector3[], data: ClusterCentersData) => {
        const sizeCoef = data.size_coef;

        for (let i = 0; i < this.nClusterCenters; i++) {
            const sprite = this.sprites[i];
            const material = sprite.material as THREE.SpriteMaterial;
            const projection = data.projections[i];

            sprite.position.copy(clusterPositions[i]);

            const isCurrent = i == data.current_id;
            const currentCoef = sizeCoef * (isCurrent ? data.current_size_coef : 1);
            this.textureLoader.load(projection,
            (texture: THREE.Texture) => {
                texture.minFilter = THREE.NearestFilter;
                texture.magFilter = THREE.NearestFilter;
                material.map = texture;
                material.needsUpdate = true;
                const ratio = texture.image.width / texture.image.height;
                sprite.scale.set(currentCoef * ratio, currentCoef, currentCoef);
            });
        }
    }

    private static buildClusterCenterSphere(): THREE.Mesh {
        const materialParams: THREE.MeshLambertMaterialParameters = { color: ClusterCenters.ShapeColor };
        const material = new THREE.MeshLambertMaterial(materialParams)
        const sphereMesh = new THREE.Mesh(ClusterCenters.SphereGeometry, material);
        return sphereMesh;
    }

    private static buildClusterCenterSprite(): THREE.Sprite {
        const materialParams: THREE.SpriteMaterialParameters = { color: 0xffffff };
        const spriteMaterial = new THREE.SpriteMaterial(materialParams);
        const sprite = new THREE.Sprite(spriteMaterial);
        return sprite;
    }

    private static buildShapeGroup(count: number, shapeFn:(() => THREE.Mesh | THREE.Sprite)): THREE.Group {
        const group = new THREE.Group();

        for (let i = 0; i < count; i++) {
            const shape = shapeFn();
            group.add(shape);
        }

        return group;
    }
}