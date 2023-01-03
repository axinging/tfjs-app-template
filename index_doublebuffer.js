function getURLState(url) {
  let params = new URLSearchParams(url);
  const keys = [...params.keys()];
  if (keys.length === 0) return null;
  let parallel = false;
  let batch = 0;
  let bufferCount = 0;
  if (params.has('parallel')) {
    parallel = params.get('parallel') == 'true' ? true : false;
  }
  if (params.has('buffer')) {
    bufferCount = Number(params.get('buffer'));
  }
  if (params.has('batch')) {
    batch = Number(params.get('batch'));
  }
  return [parallel, bufferCount, batch];
}

function predictFunction(input) {
  return model => model.predict(input);
}

async function warmup(model, predict) {
  for (let i = 0; i < 50; i++) {
    const result1 = predict(model);
    const promiseRes1 = await result1.data();
  }
}

const TEST_COUNT = 150;

async function modelDemo(model, predict) {
  const times = [];
  const numRuns = TEST_COUNT;
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

async function singleBufferDemo(model, predict) {
  let times = [];
  const WORK_PER_LOOP = 2;
  for (let i = 0; i < TEST_COUNT / WORK_PER_LOOP; i++) {
    let start = performance.now();
    const result1 = predict(model);
    const promiseRes1 = await result1.data();

    const result2 = predict(model);
    const promiseRes2 = await result2.data();
    let end = performance.now();
    times.push(Number((end - start).toFixed(2)));
  }
  const averageTime = times.reduce((acc, curr) => acc + curr, 0) /
      (times.length * WORK_PER_LOOP);
  console.log(times + ', single buffer averageTime: ' + averageTime.toFixed(2));
}

async function doubleBufferDemo(model, predict) {
  const WORK_PER_LOOP = 2;
  let promiseRes = new Array(WORK_PER_LOOP);
  let times = [];
  for (let i = 0; i < TEST_COUNT / WORK_PER_LOOP; i++) {
    let start = performance.now();
    await promiseRes[promiseRes.length - 1];
    const result1 = predict(model);
    promiseRes[0] = result1.data();

    const result2 = predict(model);
    promiseRes[1] = result2.data();
    await promiseRes[0];
    let end = performance.now();
    times.push(Number((end - start).toFixed(2)));
  }
  const averageTime = times.reduce((acc, curr) => acc + curr, 0) /
      (times.length * WORK_PER_LOOP);
  console.log(times + ', double buffer averageTime: ' + averageTime.toFixed(2));
}

async function tripleBufferDemo(
    model, predict, bufferCount, runs, comment = '') {
  let promiseRes = new Array(bufferCount);
  let times = [];
  for (let i = 0; i < runs; i++) {
    let start = performance.now();
    await promiseRes[promiseRes.length - 1];
    for (let j = 0; j < bufferCount; j++) {
      const result1 = predict(model);
      promiseRes[j] = result1.data();
      // tf.dispose(result1);
    }
    for (let j = 0; j < bufferCount - 1; j++) {
      await promiseRes[j];
    }
    let end = performance.now();
    times.push(Number((end - start).toFixed(2)));
  }
  const averageTime =
      times.reduce((acc, curr) => acc + curr, 0) / (times.length * bufferCount);
  console.log(
      times + `, ${comment} ${bufferCount} buffers: ` + averageTime.toFixed(2));
}

async function main() {
  const benchmark = benchmarks['MobileNetV3'];  // MobileNetV3,DeepLabV3
  const model = await benchmark.load();
  const predict = benchmark.predictFunc();
  await warmup(model, predict);
  const [parallel, bufferCount, batch] = getURLState(location.search);
  try {
    tf.env().set('WEBGPU_PARALLEL_COMPILATION_PASS', parallel);
  } catch {
    console.warn('WEBGPU_PARALLEL_COMPILATION_PASS is not defined');
  }

  try {
    if (batch != 0) tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', batch);
  } catch {
    console.warn('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE is not defined');
  }
  console.log(
      'parallel is : ' + parallel + ', buffer count: ' + bufferCount +
      ', batch : ' + batch);

  if (bufferCount == 1)
    await singleBufferDemo(model, predict);
  else if (bufferCount == 0)
    await modelDemo(model, predict);
  else
    await tripleBufferDemo(
        model, predict, bufferCount, TEST_COUNT / bufferCount);
}
