function predictFunction(input) {
  return model => model.predict(input);
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
    promiseRes3 = result2.data();
    await promiseRes1;
    await promiseRes2;
    let end = performance.now();
    times.push(Number((end - start).toFixed(2)));
  }
  const averageTime =
      times.reduce((acc, curr) => acc + curr, 0) / (times.length * 3);
  console.log(times + ', triple buffer averageTime: ' + averageTime.toFixed(2));
}

async function warmup(model, predict) {
  for (let i = 0; i < 50; i++) {
    // Warmup
    const result1 = predict(model);
    const promiseRes1 = await result1.data();
  }
}

async function main() {
  const benchmark = benchmarks['FaceDetection'];  // DeepLabV3
  const model = await benchmark.load();
  const predict = benchmark.predictFunc();
  await warmup(model, predict);
  await tripleBufferDemo(model, predict);
  await doubleBufferDemo(model, predict);
  await singleBufferDemo(model, predict);
}
