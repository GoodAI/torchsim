import { Matrix, SVD, ISVDOptions } from 'ml-matrix';
import { mean, standardDeviation } from 'ml-stat/matrix';

export interface AsyncProjectionCalculationInput {
    dataset: Matrix;
    maxDims: number;
}

export interface AsyncProjectionCalculationResult {
    means: number[];
    stdDevs: number[];
    U: Matrix;
    twoDimExplainedVar: number;
    threeDimExplainedVar: number;
} 

function normalize(data: Matrix, means: number[], stdDevs: number[]): Matrix {
    const normalizedData = new Matrix(data);
    normalizedData.subRowVector(means);
    normalizedData.divRowVector(stdDevs);
    return normalizedData
}

function calculatePca(input: AsyncProjectionCalculationInput): AsyncProjectionCalculationResult {
    const { dataset, maxDims } = input;
    const means = mean(dataset);
    // Zero stdDev means all values equal => centering will zero them anyways => replace the 0 with 1
    const stdDevs: number[] = standardDeviation(dataset, means, true).map(value => { return value > 0 ? value : 1; });
    const normalizedDataset = normalize(dataset, means, stdDevs);

    const svdOptions: ISVDOptions = {
        autoTranspose: true,
        computeLeftSingularVectors: false,
        computeRightSingularVectors: true,
    }

    const svd = new SVD(normalizedDataset, svdOptions); // Heavy computation right here.
    const U = svd.rightSingularVectors.subMatrix(0, normalizedDataset.columns - 1, 0, maxDims - 1);

    // Calculate explained variance.
    const singularValues = svd.diagonal;
    const eigenValues = singularValues.map((s: number) => s * s / (normalizedDataset.rows - 1));
    let twoDimExplainedVar = 0;
    let threeDimExplainedVar = 0;
    let sum = 0;

    for (let i = 0; i < eigenValues.length; i++) {
        if (i == 2) {
            twoDimExplainedVar = sum;
        } else if (i == 3) {
            threeDimExplainedVar = sum;
        }
        sum += eigenValues[i];
    }

    twoDimExplainedVar *= 100 / sum;
    threeDimExplainedVar *= 100 / sum;

    return {
        means: means,
        stdDevs: stdDevs,
        U: U,
        twoDimExplainedVar: twoDimExplainedVar,
        threeDimExplainedVar: threeDimExplainedVar,
    };
}

// The worker function itself.
const ctx: Worker = self as any;
ctx.onmessage = (event: MessageEvent) => {
    const input = event.data as AsyncProjectionCalculationInput;
    const result = calculatePca(input);
    ctx.postMessage(result);
};
