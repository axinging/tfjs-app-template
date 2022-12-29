function predictFunction(input) {
  return model => model.predict(input);
}

const benchmarks2 = {
  'MobileNetV3': {
    type: 'GraphModel',
    architectures: ['small_075', 'small_100', 'large_075', 'large_100'],
    load: async (inputResolution = 224, modelArchitecture = 'small_075') => {
      const url = `https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_${
          modelArchitecture}_224/classification/5/default/1`;
      return tf.loadGraphModel(url, {fromTFHub: true});
    },
    predictFunc: () => {
      const input = tf.randomNormal([1, 224, 224, 3]);
      return predictFunction(input);
    },
  }
};


async function doubleBufferDemo(model, predict) {
  let promiseRes1, promiseRes2;
  let times = [];
  for (let i = 0; i < 50; i++) {
    let start = performance.now();
    await promiseRes2;
    const result1 = predict(model);
    promiseRes1 = result1.data();

    const result2 = predict(model);
    promiseRes2 = result2.data();
    await promiseRes1;
    let end = performance.now();
    times.push(Number((end - start).toFixed(2)));
  }
  const averageTime = times.reduce((acc, curr) => acc + curr, 0) / times.length;
  console.log(times + ', double buffer averageTime: ' + averageTime.toFixed(2));
}

async function singleBufferDemo(model, predict) {
  let times = [];
  for (let i = 0; i < 50; i++) {
    let start = performance.now();
    const result1 = predict(model);
    const promiseRes1 = await result1.data();

    const result2 = predict(model);
    const promiseRes2 = await result2.data();
    let end = performance.now();
    times.push(Number((end - start).toFixed(2)));
  }
  const averageTime = times.reduce((acc, curr) => acc + curr, 0) / times.length;
  console.log(times + ', single buffer averageTime: ' + averageTime.toFixed(2));
}

async function warmup(model, predict) {
  for (let i = 0; i < 50; i++) {
    // Warmup
    const result1 = predict(model);
    const promiseRes1 = await result1.data();
  }
}

async function main() {
  const benchmark = benchmarks2['MobileNetV3'];
  const model = await benchmark.load();
  const predict = benchmark.predictFunc();
  await warmup(model, predict);
  await doubleBufferDemo(model, predict);
  await singleBufferDemo(model, predict);
}
