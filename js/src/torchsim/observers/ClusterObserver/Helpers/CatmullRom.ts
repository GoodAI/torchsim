// import * as THREE from 'three';
import { Vector3 } from 'three';
import { InvalidStateException } from '../../../exceptions';

export class CatmullRom {
    // private splineCoefs: number[][];
    // private tangentCoefs: number[][];

    private p0: Vector3;
    private p1: Vector3;
    private p2: Vector3;
    private p3: Vector3;
    private pointsSet: boolean = false;

    constructor() {

    }

    // constructor(nPoints: number) {

    //     this.splineCoefs = Array.from({length: nPoints}, () => );
    // }

    public setPoints = (p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3) => {
        this.p0 = p0;
        this.p1 = p1;
        this.p2 = p2;
        this.p3 = p3;
        this.pointsSet = true;
    }
 
    public spline = (t: number, out: Vector3) => { 
        if (!this.pointsSet) {
            throw new InvalidStateException(`CatmullRom.spline called without points being set.`);
        }

        const t2 = t * t; 
        const t3 = t2 * t; 
        const c0 = -t3 + 2*t2 - t;
        const c1 = 3*t3 - 5*t2 + 2;
        const c2 = -3*t3 + 4*t2 + t;
        const c3 = t3 - t2;

        this.combine(c0, c1, c2, c3, out);
 
        // return 0.5 * ((-t3 + 2*t2 - t)*p0 + (3*t3 - 5*t2 + 2)*p1 + (-3*t3 + 4*t2 + t)*p2 + (t3 - t2)*p3); 
    } 

    public tangent = (t: number, out: Vector3) => { 
        if (!this.pointsSet) {
            throw new InvalidStateException(`CatmullRom.tangent called without points being set.`);
        }

        const t2 = t * t; 
        const c0 = -3*t2 + 4*t - 1;
        const c1 = 9*t2 - 10*t;
        const c2 = -9*t2 + 8*t + 1;
        const c3 = 3*t2 - 2*t;

        this.combine(c0, c1, c2, c3, out);
 
        // return 0.5 * ((-3*t2 + 4*t - 1)*p0 + (9*t2 - 10*t)*p1 + (-9*t2 + 8*t + 1)*p2 + (3*t2 - 2*t)*p3); 
    } 

    private combine = (c0: number, c1: number, c2: number, c3: number, out: Vector3) => {
        out.copy(this.p0).multiplyScalar(c0);
        out.addScaledVector(this.p1, c1);
        out.addScaledVector(this.p2, c2);
        out.addScaledVector(this.p3, c3);
        out.multiplyScalar(0.5);
    }
}