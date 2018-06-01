/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as tf from '../index';
import {describeWithFlags} from '../jasmine_util';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, expectArraysClose, WEBGL_ENVS} from '../test_util';

describeWithFlags('scan scan1d', ALL_ENVS, () => {
  it('cumsum over 1d array', () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8]);
    const result = tf.scan(input, 'x + y');
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, [0, 1, 3, 6, 10, 15, 21, 28]);
  });

  it('comparison with tf.cumsum 1d', () => {
    const input = tf.linspace(1, 8, 8);
    const result = tf.scan(input, 'x + y');
    const canonicalResult = tf.cumsum(input, 0, true);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, canonicalResult);
  });

  it('comparison with tf.cumsum 1d non-power of two', () => {
    const input = tf.linspace(1, 8, 17);
    const result = tf.scan(input, 'x + y');
    const canonicalResult = tf.cumsum(input, 0, true);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, canonicalResult);
  });

  it('comparison with tf.cumsum 1d inclusive', () => {
    const input = tf.linspace(1, 8, 8);
    const result = tf.scan(input, 'x + y', 0, false);
    const canonicalResult = tf.cumsum(input, 0, false);
    console.log(
        tf.cumsum(input, 0, false).dataSync().toString(),
        tf.cumsum(input, 0, true).dataSync().toString());
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, canonicalResult);
  });

  it('cummul over 1d array', () => {
    const input = tf.tensor1d([1, 2, 3, 4, 5, 6, 7, 8]);
    const result = tf.scan(input, 'x * y', 1.0);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, [1, 1, 2, 6, 24, 120, 720, 5040]);
  });
});

describeWithFlags('scan scan2d', WEBGL_ENVS, () => {
  it('cumsum over 2d array', () => {
    const input =
        tf.tensor2d([[1, 2, 3, 4, 5, 6, 7, 8], [101, 2, 3, 4, 5, 6, 7, 8]]);
    const result = tf.scan(input, 'x + y');
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, tf.tensor2d([
      [0, 1, 3, 6, 10, 15, 21, 28], [0, 101, 103, 106, 110, 115, 121, 128]
    ]));
  });

  it('comparison with tf.cumsum 2d', () => {
    const input = tf.linspace(1, 8, 8);
    const result = tf.scan(input, 'x + y');
    const canonicalResult = tf.cumsum(input, 0, true);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, canonicalResult);
  });
});

describeWithFlags('scan scan2d', WEBGL_ENVS, () => {
  it('cumsum over 2d array', () => {
    const input =
        tf.tensor2d([[1, 2, 3, 4, 5, 6, 7, 8], [101, 2, 3, 4, -5, 6, 7, 8]]);
    const result = tf.scan(input, 'x + y');
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, tf.tensor2d([
      [0, 1, 3, 6, 10, 15, 21, 28], [0, 101, 103, 106, 110, 105, 111, 118]
    ]));
  });

  it('comparison with tf.cumsum 2d', () => {
    const input = tf.linspace(1, 128, 128).reshape([2, 64]);
    const result = tf.scan(input, 'x + y');
    const canonicalResult = tf.cumsum(input, 1, true);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, canonicalResult);
  });

  it('cummul over 2d array', () => {
    const input =
        tf.tensor2d([[1, 2, 3, 4, 5, 6, 7, 8], [-1, 2, 3, 4, -5, 6, 7, 8]]);
    const result = tf.scan(input, 'x * y', 1.0);
    expect(result.shape).toEqual(input.shape);
    expectArraysClose(result, tf.tensor2d([
      [1, 1, 2, 6, 24, 120, 720, 5040], [1, -1, -2, -6, -24, 120, 720, 5040]
    ]));
  });
});
