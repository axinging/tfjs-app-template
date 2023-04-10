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

function getURLState(url) {
    let params = new URLSearchParams(url);
    const keys = [...params.keys()];
    if (keys.length === 0) return '';
    let printShaderString = '';
    if (params.has('WEBGPU_PRINT_SHADER')) {
      printShaderString = params.get('WEBGPU_PRINT_SHADER');
    }
    return printShaderString;
}

async function modelDemo(model, predict) {
    const result1 = predict(model);
    const promiseRes1 = await result1.data();
    console.log(promiseRes1);
}

async function main() {
  const benchmark = benchmarks2['MobileNetV3'];
  const model = await benchmark.load();
  await tf.setBackend('webgpu');
  await tf.ready();
  const re = getURLState(location.search);
  tf.env().set('WEBGPU_PRINT_SHADER', re);
  console.log(tf.env().get('WEBGPU_PRINT_SHADER'));
  const predict = benchmark.predictFunc();
  await modelDemo(model, predict);
}
