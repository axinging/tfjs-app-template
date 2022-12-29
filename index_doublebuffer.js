function predictFunction(input) {
  return model => model.predict(input);
}

async function warmup(model, predict) {
  for (let i = 0; i < 50; i++) {
    const result1 = predict(model);
    const promiseRes1 = await result1.data();
  }
}

async function modelDemo(model, predict) {
  const times = [];
  const numRuns = 50;
  for (let i = 0; i < numRuns; i++) {
    let start = performance.now();
    console.log(performance.now().toFixed(2));
    const result1 = predict(model);
    const promiseRes1 = await result1.data();
    const elapsedTime = performance.now() - start;

    tf.dispose(result1);
    times.push(elapsedTime);
  }

  const averageTime = times.reduce((acc, curr) => acc + curr, 0) / times.length;
  console.log('average: ' + averageTime);
}

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
  const averageTime =
      times.reduce((acc, curr) => acc + curr, 0) / (times.length * 2);
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
  const averageTime =
      times.reduce((acc, curr) => acc + curr, 0) / (times.length * 2);
  console.log(times + ', single buffer averageTime: ' + averageTime.toFixed(2));
}

async function tripleBufferDemo(model, predict) {
  let promiseRes1, promiseRes2, promiseRes3;
  let times = [];
  for (let i = 0; i < 50; i++) {
    let start = performance.now();
    await promiseRes3;
    const result1 = predict(model);
    promiseRes1 = result1.data();

    const result2 = predict(model);
    promiseRes2 = result2.data();

    const result3 = predict(model);
    promiseRes3 = result3.data();
    await promiseRes1;
    await promiseRes2;
    let end = performance.now();
    times.push(Number((end - start).toFixed(2)));
  }
  const averageTime =
      times.reduce((acc, curr) => acc + curr, 0) / (times.length * 3);
  console.log(times + ', triple buffer averageTime: ' + averageTime.toFixed(2));
}

let parallel = false;
let bufferCount = 0;
function getURLState(url) {
  let params = new URLSearchParams(url);
  const keys = [...params.keys()];
  if (keys.length === 0) return null;
  let parallel = false;
  if (params.has('parallel')) {
    parallel = params.get('parallel') == 'true' ? true : false;
  }
  if (params.has('buffer')) {
    bufferCount = Number(params.get('buffer'));
  }
  return;
}

async function main() {
  const benchmark = benchmarks['MobileNetV3'];
  const model = await benchmark.load();
  const predict = benchmark.predictFunc();
  await warmup(model, predict);
  const parallel = getURLState(location.search);
  try {
    tf.env().set('WEBGPU_PARALLEL_COMPILATION_PASS', parallel);
  } catch {
    console.warn('WEBGPU_PARALLEL_COMPILATION_PASS is not defined');
  }
  console.log('parallel is : ' + parallel + ', buffer count: ' + bufferCount);

  if (bufferCount == 3) await tripleBufferDemo(model, predict);
  if (bufferCount == 2) await doubleBufferDemo(model, predict);
  if (bufferCount == 1) await singleBufferDemo(model, predict);
  if (bufferCount == 0) await modelDemo(model, predict);
}
