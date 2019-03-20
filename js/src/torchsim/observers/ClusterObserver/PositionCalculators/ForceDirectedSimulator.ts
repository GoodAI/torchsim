import { Vector3 } from 'three';
import { PositionCalculator } from './PositionCalculator';
import { InvalidArgumentException } from '../../../exceptions';
import { DataProvider } from '../Helpers/DataProvider';


export interface ForceDirectedSimulatorData {
    iterations_per_step: number;
    reset: boolean;
}

interface ForceDirectedSimulationPoint {
    force: Vector3;
    pos: Vector3;
    velocity: Vector3;
    mass: number;
}

export class ForceDirectedSimulator implements PositionCalculator {

    // Simulation settings
    private static readonly ForceCoef = 10;
    private static readonly Dt: number = 0.001;
    // private nIterations: number = 10;
    private static readonly DampeningCoef: number = Math.pow(0.975, ForceDirectedSimulator.Dt / 0.001);
    private static readonly Softening: number = 0.001;
    private static readonly CenterSpringCoef: number = 100;
    private static readonly CenterSpringLength: number = 0;
    private static readonly SpringsCoef: number = 75 * ForceDirectedSimulator.ForceCoef;
    private static readonly SpringsLength: number = 0;
    private static readonly CoulombCoef: number = 150 * ForceDirectedSimulator.ForceCoef;
    private static readonly MinConnectionStrength: number = 0.05;

    private clusterPositions: Vector3[];
    private simulatedPoints: ForceDirectedSimulationPoint[];
    private nDims: number;

    constructor(nPoints: number) {
        this.clusterPositions = [];
        this.simulatedPoints = [];
        this.nDims = 2;

        for (let i = 0; i < nPoints; i++) {
            const simulatedPoint = ForceDirectedSimulator.buildSimulatedPoint();
            this.simulatedPoints.push(simulatedPoint);
            this.clusterPositions.push(new Vector3());
        }

        this.resetSimulation()
    }

    public update = (dataProvider: DataProvider): Vector3[] => {
        const data = dataProvider.coData.fdsim;
        const clusterSimilarities = dataProvider.clusterSimilarities;
        if (data == undefined)
            throw new InvalidArgumentException(`${ForceDirectedSimulator.name}.update called without data being set.`);
        
        if (clusterSimilarities) {
            this.updateDimensionality(dataProvider.coData.n_dims);

            if (data.reset)
                this.resetSimulation();

            // Simulate
            for (let i = 0; i < data.iterations_per_step; i++) {
                this.simulationStep(clusterSimilarities);
            }

            // Copy resulting positions to output (and scale them)
            this.clusterPositions.forEach((p: Vector3, i: number) => p.copy(this.simulatedPoints[i].pos).multiplyScalar(10));
        }

        return this.clusterPositions;
    }

    private static buildSimulatedPoint = (): ForceDirectedSimulationPoint => {
        return {
            force: new Vector3(),
            pos: new Vector3(),
            velocity: new Vector3(),
            mass: 1,
        };
    }

    private resetSimulation = () => {
        const randomPoint = () => {
            const radius = 0.25;
            const theta = Math.random() * 2 * Math.PI;
            const phi = this.nDims == 2 ? 0 : Math.random() * 2 * Math.PI;

            return new Vector3(
                radius * Math.cos(theta) * Math.cos(phi),
                radius * Math.sin(theta) * Math.cos(phi),
                radius * Math.sin(phi)
            )
        }

        this.simulatedPoints.forEach(p => {
            p.pos = randomPoint();
            p.velocity.set(0, 0, 0);
            p.force.set(0, 0, 0);
        });
    }

    private updateDimensionality = (nDimsNew: number) => {
        if (nDimsNew != this.nDims) {
            let updateFunc: (() => number);
            if (nDimsNew == 2) {
                updateFunc = () => 0;
            } else {
                const initMaxAbs = 0.25;
                updateFunc = () => (Math.random() - 0.5) * 2 * initMaxAbs;
            }

            this.simulatedPoints.forEach(p => {
                p.pos.setZ(updateFunc());
                p.force.setZ(0);
                p.velocity.setZ(0);
            });
            this.nDims = nDimsNew
        }
    }

    private updateForces = (id: number, connectionStrengths: number[][]) => {
        const { force, pos, mass } = this.simulatedPoints[id];
        const dist_to_center = Math.sqrt(pos.lengthSq() + ForceDirectedSimulator.Softening);

        // Hooke's law (spring to center for every point)
        const spring_coef_global = ForceDirectedSimulator.CenterSpringCoef * (ForceDirectedSimulator.CenterSpringLength / dist_to_center - 1);
        force.copy(pos).multiplyScalar(spring_coef_global);

        for (let i = 0; i < this.simulatedPoints.length; i++)
        {
            const otherPoint = this.simulatedPoints[i];
            const posDiff = pos.clone().sub(otherPoint.pos);
            const dist = Math.sqrt(posDiff.lengthSq() + ForceDirectedSimulator.Softening);

            // Coulomb's law
            // force += diff * coulomb_coef * mass * otherPoint.mass / (dist * dist * dist);
            const repulsion_coef = ForceDirectedSimulator.CoulombCoef * mass * otherPoint.mass / (dist * dist * dist);
            force.addScaledVector(posDiff, repulsion_coef);

            const conn_strength = Math.max(connectionStrengths[id][i], connectionStrengths[i][id]);
            if (conn_strength >= ForceDirectedSimulator.MinConnectionStrength) {
                // Hooke's law (InternalNodes repo, 422e0b66bb95214a414d451b7ad3554cf0fa3cb0, FDObserver.cs:229 for original form)
                const spring_coef = ForceDirectedSimulator.SpringsCoef * conn_strength * (ForceDirectedSimulator.SpringsLength / dist - 1);
                force.addScaledVector(posDiff, spring_coef);
            }
        }
    }

    private simulationStep = (connectionStrengths: number[][]) => {
        const dt = ForceDirectedSimulator.Dt;
        const dtHalf = dt / 2;
        const dt2Half = dt * dt / 2;

        for (let i = 0; i < this.simulatedPoints.length; i++) {
            const point = this.simulatedPoints[i];
            const {force, pos, velocity, mass} = point;

            {
                // Newton's 2nd law: acc = F/m
                const acc = force.clone().divideScalar(mass);

                // Verlet integration (update locations)
                // pos += velocity * dt + acc * dt2 * 0.5;
                pos.addScaledVector(velocity, dt);
                pos.addScaledVector(acc, dt2Half);

                // Verlet integration (first half update velocity)
                // velocity += acc * dt * 0.5;
                velocity.addScaledVector(acc, dtHalf);
            }

            this.updateForces(i, connectionStrengths);

            {
                // Compute acc again as force has changed.
                const acc = force.clone().divideScalar(mass);

                // Verlet integration (second half update velocity)
                // velocity += acc * dt * 0.5;
                velocity.addScaledVector(acc, dtHalf);

                // Dampen speed by coef
                velocity.multiplyScalar(ForceDirectedSimulator.DampeningCoef);
            }
        }
    }
}