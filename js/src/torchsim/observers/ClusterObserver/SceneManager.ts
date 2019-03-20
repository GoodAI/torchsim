import * as THREE from 'three';
import { InvalidArgumentException } from '../../exceptions';

import { AxesHelper } from './SceneSubjects/AxesHelper';
import { ClusterCenters } from './SceneSubjects/ClusterCenters';
import { ClusterObserverData } from '../ClusterObserver';
import { ForceDirectedSimulator } from './PositionCalculators/ForceDirectedSimulator';
import { TrackballControls } from './Helpers/TrackballControls'
import { PcaProjector } from './PositionCalculators/PcaProjector';
import { PositionCalculator } from './PositionCalculators/PositionCalculator';
import { SceneSubject } from './SceneSubjects/SceneSubject';
import { ClusterDatapoints } from './SceneSubjects/ClusterDatapoints';
import { SplineArrows } from './SceneSubjects/SplineArrows';
import { SpringLines } from './SceneSubjects/SpringLines';
import { DataProvider } from './Helpers/DataProvider';


interface CanvasDimensions {
    width: number;
    height: number;
}

interface SceneSubjects {
    axesHelper?: AxesHelper;
    clusterCenters?: ClusterCenters;
    clusterDatapoints?: ClusterDatapoints;
    splineArrows?: SplineArrows;
    springLines?: SpringLines;
}

export class SceneManager {
    private canvasDimensions: CanvasDimensions;

    private currentCamera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
    private camera2D: THREE.PerspectiveCamera | THREE.OrthographicCamera;
    private camera3D: THREE.PerspectiveCamera;
    private cameraControls: THREE.TrackballControls;
    private scene: THREE.Scene;
    private renderer: THREE.WebGLRenderer;

    private positionCalculator: PositionCalculator;
    private dataProvider: DataProvider;
    private sceneSubjects: SceneSubjects;

    constructor(canvas: HTMLCanvasElement, coData: ClusterObserverData) {
        this.canvasDimensions = { width: canvas.width, height: canvas.height };

        this.camera2D = SceneManager.buildCameraOrthographic(this.canvasDimensions);
        // this.camera2D = SceneManager.buildCameraPerspective(this.canvasDimensions);
        this.camera3D = SceneManager.buildCameraPerspective(this.canvasDimensions);
        this.currentCamera = this.camera3D;
        this.cameraControls = SceneManager.buildTrackballControls(this.camera3D, canvas, this.render);
        this.scene = SceneManager.buildScene();
        this.renderer = SceneManager.buildRenderer(canvas, this.canvasDimensions);

        this.dataProvider = new DataProvider();
        this.sceneSubjects = {};

        this.render();
    }

    public onCanvasResize = (width: number, height: number) => {
        this.canvasDimensions.width = width;
        this.canvasDimensions.height = height;

        // this.camera2D gets recalculated to fit bbox anyways, so not updating it here.
        this.camera3D.aspect = width / height;
        this.camera3D.updateProjectionMatrix();

        // TODO: setSize changes size of the canvas again, should use something else (like setViewport?)
        this.renderer.setSize(width, height);
        this.cameraControls.handleResize();
    };

    public onMouseMove = (dx: number, dy: number) => {
        // this.camera.position.x += dx * 5;
        // this.camera.position.y += dy * 5;
        // this.camera.lookAt(0, 0, 0);
    }

    public update = (coData: ClusterObserverData) => {
        this.checkCanvasSize(coData);
        this.checkPositionCalculator(coData);
        this.checkSceneSubjects(coData);

        this.dataProvider.update(coData);
        const positions = this.positionCalculator.update(this.dataProvider);
        this.forEachSceneSubject(subject => subject.update(positions, this.dataProvider));
        this.checkCamera(positions, coData);
    };

    public render = () => {
        this.renderer.render(this.scene, this.currentCamera);
    };

    private static makeCameraFit(camera: THREE.PerspectiveCamera | THREE.OrthographicCamera, boundingBox: THREE.Box3, canvasAspect: number) {
        const bbox = boundingBox.clone();
        const center = new THREE.Vector3();

        // Add border
        {
            const borderCoef = 0.15;
            const minBorderMargin = 2.5;
            const width = bbox.max.x - bbox.min.x;
            const height = bbox.max.y - bbox.min.y;

            const borderWidth = Math.max(width / 2 * borderCoef, minBorderMargin);
            bbox.max.x += borderWidth;
            bbox.min.x -= borderWidth;

            const borderHeight = Math.max(height / 2 * borderCoef, minBorderMargin);
            bbox.max.y += borderHeight;
            bbox.min.y -= borderHeight;
        }

        // Compensate for canvas aspect ratio
        {
            const width = bbox.max.x - bbox.min.x;
            const height = bbox.max.y - bbox.min.y;
            const aspect = width / height;

            // Shift bbox to center so that all transformations are symmetrical.
            bbox.getCenter(center);
            bbox.min.sub(center);
            bbox.max.sub(center);

            if (canvasAspect >= aspect) {
                // halfWidth = halfHeight * canvasAspect;
                bbox.min.x = bbox.min.y * canvasAspect;
                bbox.max.x = bbox.max.y * canvasAspect;
            } else {
                // halfHeight = halfWidth / canvasAspect;
                bbox.min.y = bbox.min.x / canvasAspect;
                bbox.max.y = bbox.max.x / canvasAspect;
            }

            // And shift it back afterwards.
            bbox.min.add(center);
            bbox.max.add(center);
        }

        if (camera instanceof THREE.PerspectiveCamera) {
            const width = bbox.max.x - bbox.min.x;
            const fov = camera.fov * Math.PI / 180;
            const dist = width / (2 * Math.tan(fov / 2) * canvasAspect);
            camera.position.set(center.x, center.y, dist);
        } else if (camera instanceof THREE.OrthographicCamera) {
            const halfDepth = 2500;
            Object.assign(camera, {
                left: bbox.min.x,
                right: bbox.max.x,
                top: bbox.max.y,
                bottom: bbox.min.y,
                near: 0,
                far: 2 * halfDepth,
            });
            camera.updateProjectionMatrix();
            camera.position.z = halfDepth;
        }
    }

    private static buildCameraOrthographic({ width, height }: CanvasDimensions): THREE.OrthographicCamera {
        const aspect = width / height;
        const halfDepth = 10;
        const halfHeight = 10;

        const camera = new THREE.OrthographicCamera(-halfHeight * aspect, halfHeight * aspect, halfHeight, -halfHeight, 0, 2 * halfDepth);
        camera.position.z = halfDepth;

        return camera;
    }

    private static buildCameraPerspective({ width, height }: CanvasDimensions): THREE.PerspectiveCamera {
        const aspectRatio = width / height;
        const fieldOfView = 45;
        const nearPlane = 0.1;
        const farPlane = 100000;

        const camera = new THREE.PerspectiveCamera(fieldOfView, aspectRatio, nearPlane, farPlane);
        camera.position.z = 60;

        return camera;
    }

    private static buildTrackballControls(camera: THREE.Camera, canvas: HTMLCanvasElement, onChange: (() => void)): THREE.TrackballControls {
        const controls = new TrackballControls(camera, canvas);
        Object.assign(controls, { panSpeed: 0.9, rotateSpeed: 2.5, zoomSpeed: 2.0, staticMoving: true });
        controls.addEventListener('change', onChange);
        return controls;
    }

    private static buildRenderer(canvas: HTMLCanvasElement, { width, height }: CanvasDimensions) {
        const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true})
        const pixelRatio = window.devicePixelRatio ? window.devicePixelRatio : 1;

        renderer.setPixelRatio(pixelRatio);
        // TODO: setSize changes size of the canvas again, should use something else (like setViewport?)
        renderer.setSize(width, height);
        renderer.setClearColor(0xff0000);

        return renderer;
    }

    private static buildScene(): THREE.Scene {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
        scene.add(ambientLight);
        const light = new THREE.DirectionalLight(0xffffff, 0.6);
        scene.add(light);
        const light2 = new THREE.DirectionalLight(0xffffff, 0.1);
        light2.position.setY(-1);
        light2.lookAt(0, 0, 0);
        scene.add(light2);

        return scene;
    }

    private calculateCurrentBoundingBox = (positions: THREE.Vector3[], coData: ClusterObserverData): THREE.Box3 => {
        const bbox = new THREE.Box3().setFromPoints(positions);
        if (coData.pca != undefined && coData.pca.cluster_datapoints != undefined && coData.pca.cluster_datapoints.cluster_datapoints) {
            const dataPoints = coData.pca.cluster_datapoints.cluster_datapoints;
            const tmp = new THREE.Vector3();
            for (let i = 0; i < dataPoints.length; ++i) {
                const point = dataPoints[i];
                tmp.set(point[0], point[1], point[2]);
                bbox.expandByPoint(tmp);
            }
        }

        if (this.sceneSubjects.axesHelper != undefined) {
            bbox.expandByPoint(this.sceneSubjects.axesHelper.sceneObject.position);
        }

        return bbox;
    }

    private checkCamera = (positions: THREE.Vector3[], coData: ClusterObserverData) => {
        if (coData.n_dims == 2) {
            const bbox = this.calculateCurrentBoundingBox(positions, coData);
            const canvasAspect = this.canvasDimensions.width / this.canvasDimensions.height;
            SceneManager.makeCameraFit(this.camera2D, bbox, canvasAspect);
            this.currentCamera = this.camera2D;
            this.cameraControls.enabled = false;
        } else {
            this.currentCamera = this.camera3D;
            this.cameraControls.enabled = true;
        }
    };

    private checkCanvasSize = (coData: ClusterObserverData) => {
        const { width, height } = coData;
        if (this.canvasDimensions.width !== width || this.canvasDimensions.height !== height) {
            this.onCanvasResize(width, height);
        }
    };

    private checkPositionCalculator = (coData: ClusterObserverData) => {
        const typeName = coData.projection_type;
        const count = coData.n_cluster_centers;

        if (typeName === 'FDsim') {
            if (this.positionCalculator instanceof ForceDirectedSimulator === false)
                this.positionCalculator = new ForceDirectedSimulator(count);
        } else if (typeName === 'PCA') {
            if (this.positionCalculator instanceof PcaProjector === false)
                this.positionCalculator = new PcaProjector(count);
        } else {
            throw new InvalidArgumentException(`Unknown position calculator type: '${typeName}'`);
        }
    };

    private checkSceneSubjects = (coData: ClusterObserverData) => {
        const addSceneSubject = (name: string, factoryFunc: (() => SceneSubject)) => {
            if (name in this.sceneSubjects === false) {
                const subject = factoryFunc();
                this.sceneSubjects[name] = subject;
                this.scene.add(subject.sceneObject);
            }
        };
        const removeSceneSubject = (name: keyof SceneSubjects) => {
            if (name in this.sceneSubjects) {
                const subject: SceneSubject = this.sceneSubjects[name];
                this.scene.remove(subject.sceneObject);
                delete this.sceneSubjects[name]
            }
        };
        const checkSceneSubject = (enabled: boolean, name: keyof SceneSubjects, factoryFunc: (() => SceneSubject)) => {
            if (enabled)
                addSceneSubject(name, factoryFunc);
            else
                removeSceneSubject(name);
        };

        checkSceneSubject(
            coData.n_dims == 3 || coData.projection_type == 'PCA',
            'axesHelper',
            () => new AxesHelper()
        );

        checkSceneSubject(
            coData.cluster_centers != undefined,
            'clusterCenters',
            () => new ClusterCenters(coData.n_cluster_centers)
        );

        checkSceneSubject(
            coData.pca != undefined && coData.pca.cluster_datapoints != undefined,
            'clusterDatapoints',
            () => new ClusterDatapoints(coData.n_cluster_centers)
        );

        checkSceneSubject(
            coData.spline_arrows != undefined,
            'splineArrows',
            () => new SplineArrows(coData.sequence_length, coData.n_sequences)
        );

        checkSceneSubject(
            coData.spring_lines != undefined,
            'springLines',
            () => new SpringLines(coData.n_cluster_centers)
        );
    };

    private forEachSceneSubject = (callbackfn: (value: SceneSubject) => void) => {
        Object.values(this.sceneSubjects).forEach(callbackfn)
    };
}