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

import * as handtrack from '@tensorflow-models/handtrack';
import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';


tf.ENV.set('DEBUG', true);

/**
 * Start the demo.
 */
const bindPage = async () => {
  await tf.ready();

  {
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
  }

  {
    const a = tf.tensor1d([2, 3]);
    const b = tf.tensor1d([-2, -3]);
    const c = a.mul(b);
    c.print();
  }

}

bindPage();
