/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

//import * as handtrack from '@tensorflow-models/handtrack';
//import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';


tf.ENV.set('DEBUG',true);

function generateCaseInputs(totalSizeTensor, totalSizeFilter) {
  const inp = new Array(totalSizeTensor);
  const filt = new Array(totalSizeFilter);

  for (let i = 0; i < totalSizeTensor; i++) {
    inp[i] = i;
  }
  for (let i = 0; i < totalSizeFilter; i++) {
    filt[i] = i;
  }

  return {input: inp, filter: filt};
}

/**
 * Start the demo.
 */
const bindPage = async () => {
  // await tf.ready();
  const width = 4;
  const height = 8;
  const inputDepth = 1;
  const inputShape = [1, width, height, inputDepth];
  const outputDepth = 1;
  const fSize = 3;
  const pad = 'valid';
  const stride = [1, 1];

  const inputs = generateCaseInputs(1 * width * height * inputDepth, fSize * fSize);
  const x = tf.tensor4d(inputs.input, inputShape);
  const w =
      tf.tensor4d(inputs.filter, [fSize, fSize, inputDepth, outputDepth]);

  const result = tf.conv2d(x, w, stride, pad);
  console.log(await result.data());

  /*
  // f(a, b) = a * b
  const f = (a, b) => a.mul(b);
  // df / da = b, df / db = a
  const g = tf.grads(f);

  const a = tf.tensor1d([2, 3]);
  const b = tf.tensor1d([-2, -3]);
  const [da, db] = g([a, b]);
  console.log('da');
  da.print();
  console.log('db');
  db.print();
  */
}

bindPage();
