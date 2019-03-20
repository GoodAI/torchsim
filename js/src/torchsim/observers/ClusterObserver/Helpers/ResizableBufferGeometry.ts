import * as THREE from 'three';

interface BufferAttributeOptions {
    dynamic: boolean;
    elementSize: number;
    name: string;
    normalized?: boolean;
    type: { new(count: number): Float32Array | Uint8Array };
}

export class ResizableBufferGeometry extends THREE.BufferGeometry {
    private static readonly ColorAttribOptions: BufferAttributeOptions = {
        dynamic: true,
        elementSize: 3,
        name: 'color',
        type: Float32Array,
    };

    public static readonly NormalAttribOptions: BufferAttributeOptions = {
        dynamic: true,
        elementSize: 3,
        name: 'normal',
        normalized: true,
        type: Float32Array,
    };

    public static readonly PositionAttribOptions: BufferAttributeOptions = {
        dynamic: true,
        elementSize: 3,
        name: 'position',
        type: Float32Array,
    };

    public static readonly UvAttribOptions: BufferAttributeOptions = {
        dynamic: false,
        elementSize: 2,
        name: 'uv',
        normalized: true,
        type: Float32Array,
    };

    public copyArrayToAttribute = (array: number[], options: BufferAttributeOptions) => {
        const attrib = this.ensureAttributeSize(array.length / options.elementSize, options);
        attrib.copyArray(array);
        attrib.needsUpdate = true;
    };

    public copyColorsToAttribute = (array: THREE.Color[]) => {
        const colorAttrib = this.ensureAttributeSize(array.length, ResizableBufferGeometry.ColorAttribOptions);
        colorAttrib.copyColorsArray(array);
        colorAttrib.needsUpdate = true;
    };

    public copyVector3sToAttribute = (array: THREE.Vector3[], options: BufferAttributeOptions) => {
        const attrib = this.ensureAttributeSize(array.length, options);
        attrib.copyVector3sArray(array);
        attrib.needsUpdate = true;
    };

    public setIndexAttribute = (index: number[]) => {
        super.setIndex(index)
    };

    private ensureAttributeSize = (elementCount: number, options: BufferAttributeOptions): THREE.BufferAttribute => {
        const nValues = elementCount * options.elementSize;
        const attrib = this.getAttribute(options.name) as THREE.BufferAttribute;

        if (attrib != undefined && attrib.count === nValues) {
            return attrib;
        } else {
            if (attrib != undefined) {
                this.removeAttribute(options.name);
            }

            const newAttrib = new THREE.BufferAttribute(new options.type(nValues), options.elementSize, options.normalized);
            newAttrib.dynamic = options.dynamic;
            this.addAttribute(options.name, newAttrib);
            return newAttrib;
        }
    };
}