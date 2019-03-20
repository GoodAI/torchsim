import * as THREE from 'three';
import { ResizableBufferGeometry } from '../Helpers/ResizableBufferGeometry';
import { SceneSubject } from './SceneSubject';
import { CatmullRom } from '../Helpers/CatmullRom';
import { InvalidArgumentException } from '../../../exceptions';
import { Mesh } from 'three';
import { DataProvider } from '../Helpers/DataProvider';

export interface SplineArrowsData {
    significances?: number[];
    significance_root_power: number;
    size_coef: number;
    inner_shift_coef: number;
    outer_shift_coef: number;
}

export class SplineArrows implements SceneSubject {
    public static readonly TubularSegments: number = 20;
    public static readonly RadialSegments: number = 8;
    public static readonly MinArrowSignificance: number = 0.05;

    private sequences: number[][];
    private significances: number[];
    private splineArrowsGroup: THREE.Group;

    constructor(sequenceLength: number, maxSequences: number) {
        this.splineArrowsGroup = SplineArrows.buildShapeGroup(sequenceLength, maxSequences);
    }

    private get arrows(): Arrow[] {
        return this.splineArrowsGroup.children as Arrow[];
    }

    public get sceneObject(): THREE.Object3D {
        return this.splineArrowsGroup;
    }

    public update = (clusterPositions: THREE.Vector3[], dataProvider: DataProvider) => {
        const data = dataProvider.coData.spline_arrows;

        if (data == undefined)
            throw new InvalidArgumentException(`${SplineArrows.name}.update called without data being set.`);

        this.sequences = dataProvider.coData.sequences.sequences || this.sequences;
        this.significances = data.significances || this.significances;

        if (this.sequences == undefined || this.significances == undefined)
            return;

        const max = this.significances.reduce((prevMax, val) => Math.max(prevMax, val));
        const pow = 1 / data.significance_root_power;
        const normalizedSignificances = max > 0 ? this.significances.map(value => Math.pow(value / max, pow)) : this.significances;
        const colors = DataProvider.generateColors(this.sequences.length);

        for (let i = 0; i < this.sequences.length; ++i) {
            const significance = normalizedSignificances[i];
            const arrow = this.arrows[i];

            if (significance < SplineArrows.MinArrowSignificance) {
                arrow.visible = false;
                continue;
            }

            arrow.visible = true;

            const sequence = this.sequences[i];
            const points = sequence.map(id => clusterPositions[id]);
            arrow.update(points, significance * data.size_coef, colors[i], data.inner_shift_coef, data.outer_shift_coef);
        }
    };

    private static buildSplineArrowShape(sequenceLength: number): THREE.Group {
        const arrow = new Arrow(sequenceLength);
        return arrow;
    }

    private static buildShapeGroup(sequenceLength: number, count: number): THREE.Group {
        const group = new THREE.Group();

        for (let i = 0; i < count; i++) {
            const shape = SplineArrows.buildSplineArrowShape(sequenceLength);
            group.add(shape);
        }

        return group;
    }
}

class Arrow extends THREE.Group {

    private spline: CatmullRom = new CatmullRom();
    private nPoints: number;

    // private static readonly InnerShiftCoef: number = 0.25;
    // private static readonly OuterShiftCoef: number = 0.05;

    constructor(nPoints: number) {
        super();

        if (nPoints <= 1) {
            throw new InvalidArgumentException("Arrow needs at least 2 points");
        }

        this.nPoints = nPoints;

        const materialParams: THREE.MeshLambertMaterialParameters = {
            // color: new THREE.Color('green'),
            // side: THREE.FrontSide,
            // wireframe: true,
            // map: new THREE.TextureLoader().load('https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/sprites/disc.png'),
        };
        const material = new THREE.MeshLambertMaterial(materialParams)

        for (let i = 1; i < this.nPoints; ++i) {
            const geometry = new ArrowPartBufferGeometry(SplineArrows.TubularSegments, SplineArrows.RadialSegments, i == this.nPoints - 1 ? true : false);
            const mesh = new THREE.Mesh(geometry, material);
            this.add(mesh);
        }
    }

    public update = (points: THREE.Vector3[], radius: number, color: THREE.Color, innerShiftCoef: number, outerShiftCoef: number) => {
        if (points.length != this.nPoints) {
            throw new InvalidArgumentException("Number of points doesn't match.");
        }

        points = Arrow.modify(points, innerShiftCoef, outerShiftCoef);
        let p0 = undefined;
        let p1 = points[0];
        let p2 = p1;
        let p3 = points[1];
        for (let i = 1; i < this.nPoints; ++i) {
            p0 = p1;
            p1 = p2;
            p2 = p3;
            p3 = points[Math.min(i + 1, points.length - 1)];

            this.spline.setPoints(p0, p1, p2, p3);

            const part = this.children[i - 1] as Mesh;
            (part.material as THREE.MeshPhongMaterial).color.set(color);
            (part.geometry as ArrowPartBufferGeometry).update(radius, this.spline);
        }

    }

    private static calculateInnerShifted(v1: THREE.Vector3, v2: THREE.Vector3, v3: THREE.Vector3, coef: number): THREE.Vector3 {
        // v2 + coef * ((v1-v2) + (v3-v2)) / 2
        return v1.clone().sub(v2).add(v3).sub(v2).multiplyScalar(coef / 2).add(v2);
    }

    private static calculateOuterShifted(v1: THREE.Vector3, v2: THREE.Vector3, coef: number): THREE.Vector3 {
        // (1-coef) * v1 + coef * v2
        return new THREE.Vector3().lerpVectors(v1, v2, coef);
    }

    private static modify(points: THREE.Vector3[], innerShiftCoef: number, outerShiftCoef: number): THREE.Vector3[] {
        const res: THREE.Vector3[] = [];

        let p0 = points[0];
        let p1 = p0;
        let p2 = points[1];
        let p3 = points[Math.min(2, points.length - 1)];

        res.push(Arrow.calculateOuterShifted(p1, p3, outerShiftCoef));
        res.push(Arrow.calculateInnerShifted(p1, p2, p3, innerShiftCoef));

        for (let i = 3; i < points.length; ++i) {
            p0 = p1;
            p1 = p2;
            p2 = p3;
            p3 = points[i];
            
            res.push(Arrow.calculateInnerShifted(p1, p2, p3, innerShiftCoef));
        }

        res.push(Arrow.calculateOuterShifted(p3, p1, outerShiftCoef))

        return res;
    }
}

class ArrowPartBufferGeometry extends ResizableBufferGeometry {

    // public static readonly type: string = 'ArrowPartBufferGeometry';

    private radialSegments: number;
    private tubularSegments: number;
    private arrowHeadCoef: number = 3;
    private arrowHeadStart: number = 0.8;

    private arrowNormals: number[];
    private arrowVertices: number[];

    private isArrowEnd: boolean;

    constructor(tubularSegments: number, radialSegments: number, isArrowEnd: boolean) {
        super();

        this.radialSegments = radialSegments;
        this.tubularSegments = tubularSegments;
        this.isArrowEnd = isArrowEnd;

        const additionalSegments = isArrowEnd ? 3 : 2;
        const indices = ArrowPartBufferGeometry.generateIndices(tubularSegments + additionalSegments, radialSegments);
        this.setIndexAttribute(indices);

        // const uvs = ArrowBufferGeometry.generateUVs(tubularSegments + 2, radialSegments);
        // this.copyArrayToAttribute(uvs, ResizableBufferGeometry.UvAttribOptions);
    }

    public update = (radius: number, spline: CatmullRom) => {
        this.arrowNormals = [];
        this.arrowVertices = [];

        radius /= this.arrowHeadCoef;

        if (this.isArrowEnd) {
            this.createArrowSegments(radius, spline);
        } else {
            this.createTubeSegments(radius, spline);
        }

        this.copyArrayToAttribute(this.arrowVertices, ResizableBufferGeometry.PositionAttribOptions);
        this.copyArrayToAttribute(this.arrowNormals, ResizableBufferGeometry.NormalAttribOptions);
        this.computeBoundingBox();
    };

    private createArrowSegments = (radius: number, spline: CatmullRom) => {
        const arrowStart = Math.round(this.arrowHeadStart * this.tubularSegments);

        const arrowHeadSegments = this.tubularSegments - arrowStart;

        // this.appendSegment(0, 0, spline, true);
        for (let i = 0; i <= this.tubularSegments; i++) {
            if (i < arrowStart) {
                this.appendSegment(i, radius, spline, false);
            } else if (i == arrowStart) {
                this.appendSegment(i, radius, spline, false);
                this.appendSegment(i, radius, spline, true);
                this.appendSegment(i, radius * this.arrowHeadCoef, spline, true);
                this.appendSegment(i, radius * this.arrowHeadCoef, spline, false);
            } else {
                const r = radius * this.arrowHeadCoef * (this.tubularSegments - i) / arrowHeadSegments;
                this.appendSegment(i, r, spline, false);
            }
        }
    }

    private createTubeSegments = (radius: number, spline: CatmullRom) => {
        this.appendSegment(0, 0, spline, true);
        this.appendSegment(0, radius, spline, true);
        for (let i = 0; i <= this.tubularSegments; i++) {
            this.appendSegment(i, radius, spline, false);
        }
    }

    private appendSegment = (tubular_idx: number, radius: number, spline: CatmullRom, isPerpendicular: boolean) => {
        const t = tubular_idx / this.tubularSegments;
        const arrowNormal = new THREE.Vector3();
        const arrowVertex = new THREE.Vector3();

        const v1 = new THREE.Vector3();
        const v2 = new THREE.Vector3();

        const curvePoint = new THREE.Vector3();
        const curveTangent = new THREE.Vector3();

        spline.spline(t, curvePoint);
        spline.tangent(t, curveTangent);

        if (curveTangent.x == 0 && curveTangent.y == 0) {
            v1.x = 1;
        } else {
            v1.x = -curveTangent.y;
            v1.y = curveTangent.x;
            v1.normalize();
        }

        v2.crossVectors(curveTangent, v1);
        v2.normalize();


        for (let i = 0; i <= this.radialSegments; i++) {
            const theta = i / this.radialSegments * Math.PI * 2;
            const sin = Math.sin(theta);
            const cos = -Math.cos(theta);

            arrowNormal.copy(v1).multiplyScalar(cos);
            arrowNormal.addScaledVector(v2, sin);
            arrowNormal.normalize();
            arrowVertex.copy(curvePoint).addScaledVector(arrowNormal, radius);

            if (isPerpendicular) {
                this.arrowNormals.push(-curveTangent.x, -curveTangent.y, -curveTangent.z);
            } else {
                this.arrowNormals.push(arrowNormal.x, arrowNormal.y, arrowNormal.z);
            }
            this.arrowVertices.push(arrowVertex.x, arrowVertex.y, arrowVertex.z);
        }
    }

    private static generateIndices(tubularSegments: number, radialSegments: number): number[] {
        const indices: number[] = []

        for (let j = 1; j <= tubularSegments; j++) {
            for (let i = 1; i <= radialSegments; i++) {

                const a = (radialSegments + 1) * (j - 1) + (i - 1);
                const b = (radialSegments + 1) * j + (i - 1);
                const c = (radialSegments + 1) * j + i;
                const d = (radialSegments + 1) * (j - 1) + i;

                indices.push(a, b, d);
                indices.push(b, c, d);
            }
        }

        return indices;
    }

    // TODO: account for repeated segments of sharp edges of arrows (if UVs are ever used)
    // private static generateUVs(tubularSegments: number, radialSegments: number): number[] {
    //     const uvs = [];

    //     for (let i = 0; i <= tubularSegments; i ++ ) {
    //         for (let j = 0; j <= radialSegments; j ++ ) {

    //             const u = i / tubularSegments;
    //             const v = j / radialSegments;
    //             uvs.push(u, v);
    //         }
    //     }

    //     return uvs;
    // }
}
