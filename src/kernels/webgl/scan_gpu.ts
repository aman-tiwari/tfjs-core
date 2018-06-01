/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class ScanSequentialProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(
      shape: number[], axis: number, binOpName: string, binOpDef: string,
      identity: number, exclusive: boolean) {
    this.outputShape = shape.slice();
    const rank = shape.length;
    const finalDim = shape[shape.length - 1];
    const init = identity.toString();
    this.userCode = `
      ${binOpDef}

      void main() {
        ${getCoordsDataType(rank)} coords = getOutputCoords();
        int end = ${getScanAxisCoord(rank, axis, 'coords')};
        float val = float(${init});
        for (int idx = ${finalDim} - 1; idx >= 0; idx -= 1) {
          if (idx >= end) {
            continue;
          }
          ${getScanAxisCoord(rank, axis, 'coords')} = idx;
          val = ${binOpName}(val, getX(${getCoords(rank, 'coords')}));
        }
        setOutput(val);
      }
    `;
  }
}

export class ScanContractProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(
      shape: number[], axis: number, binOpName: string, binOpDef: string) {
    this.outputShape = shape.slice();
    this.outputShape[axis] = Math.floor(this.outputShape[axis] / 2);
    const rank = shape.length;

    this.userCode = `
    ${binOpDef}
    void main() {
      ${getCoordsDataType(rank)} coords = getOutputCoords();
      // stretch out coords to index into x
      ${getScanAxisCoord(rank, axis, 'coords')} *= 2;

      float x = getX(${getCoords(rank, 'coords')});

      ${getScanAxisCoord(rank, axis, 'coords')} += 1;

      float y = getX(${getCoords(rank, 'coords')});
      setOutput(${binOpName}(x, y));
    }
    `;
  }
}

export class ScanMergeProgram implements GPGPUProgram {
  variableNames = ['x', 'r'];
  outputShape: number[];
  userCode: string;

  constructor(
      shapeX: number[], shapeR: number[], axis: number, binOpName: string,
      binOpDef: string) {
    this.outputShape = shapeX.slice();
    if (Math.floor(shapeX[axis]) / 2 !== shapeR[axis]) {
      throw new Error('Size of contracted dim must be half size of input dim');
    } else if (shapeX.length !== shapeR.length) {
      throw new Error('Tensors must be same rank');
    }

    const rank = shapeX.length;
    this.userCode = `
    ${binOpDef}

    void main() {
      // index into input & output arrays
      ${getCoordsDataType(rank)} coordsX = getOutputCoords();
      // coordsR = index into contracted result for upsweep
      // (contracted axis is half size of input axis)
      ${getCoordsDataType(rank)} coordsR = getOutputCoords();
      ${getScanAxisCoord(rank, axis, 'coordsR')} /= 2;

      if(mod(float(${getScanAxisCoord(rank, axis, 'coordsX')}), 2.0) == 0.0) {
        setOutput(getR(${getCoords(rank, 'coordsR')}));
      } else {
        ${getScanAxisCoord(rank, axis, 'coordsX')} -= 1;
        float r = getR(${getCoords(rank, 'coordsR')});
        float x = getX(${getCoords(rank, 'coordsX')});
        setOutput(${binOpName}(r, x));
      }
    }
    `;
  }
}

function getCoords(rank: number, name: string): string {
  if (rank === 1) {
    return `${name}`;
  } else if (rank === 2) {
    return `${name}.x, ${name}.y`;
  } else if (rank === 3) {
    return `${name}.x, ${name}.y, ${name}.z`;
  } else if (rank === 4) {
    return `${name}.x, ${name}.y, ${name}.z, ${name}.w`;
  } else {
    throw Error(`Cumulative sum for rank ${rank} is not yet supported`);
  }
}

function getScanAxisCoord(rank: number, axis: number, name: string): string {
  if (axis >= rank) {
    throw Error(`Can't scan on ${axis} of a ${rank}-dimensional tensor`);
  }
  if (rank === 1) {
    return `${name}`;
  } else if (rank === 2) {
    return `${name}.y`;
  } else if (rank === 3) {
    return `${name}.z`;
  } else if (rank === 4) {
    return `${name}.w`;
  } else {
    throw Error(`Scan sum for rank ${rank} is not yet supported`);
  }
}
