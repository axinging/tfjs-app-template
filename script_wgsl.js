// https://developers.google.com/web/updates/2019/08/get-started-with-gpu-compute-on-the-web
// import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {getComputeShaderCodeWGSL} from './shader_wgsl.js';

function acquireBuffer(device, byteSize, usage) {
  const newBuffer = device.createBuffer({size: byteSize, usage: usage});
  return newBuffer;
}

function makeUniforms(device, programUniform) {
  let currentOffset = 0;
  let preLength = 0;
  const offsets = [];
  let maxAlignmentOfField = 1;
  programUniform.forEach((d) => {
    if (d.data.length === 0) {
      d.data = [1];
    }
    // https://www.w3.org/TR/WGSL/#alignof
    let baseAlignment;
    switch (d.data.length) {
      case 1:
        baseAlignment = 4;
        break;
      case 2:
        baseAlignment = 8;
        break;
      case 3:
        baseAlignment = 16;
        break;
      case 4:
        baseAlignment = 16;
        break;
      case 5:
        baseAlignment = 16;
        break;
      case 6:
        baseAlignment = 16;
        break;
      default:
        util.assert(false, () => `Unsupported ${d.data.length}D shape`);
    }

    if (preLength === 5 || preLength === 6) {
      baseAlignment = 16;
    }
    if (baseAlignment > maxAlignmentOfField) {
      maxAlignmentOfField = baseAlignment;
    }
    currentOffset = Math.ceil(currentOffset / baseAlignment) * baseAlignment;
    preLength = d.data.length;
    offsets.push(currentOffset);
    currentOffset += d.data.length * 4;
  });

  currentOffset =
      Math.ceil(currentOffset / maxAlignmentOfField) * maxAlignmentOfField;
  const arrayBuffer = new ArrayBuffer(currentOffset);
  programUniform.forEach((d, i) => {
    const offset = offsets[i];
    if (d.type === 'int32') {
      new Int32Array(arrayBuffer, offset, d.data.length).set(d.data);
    } else if (d.type === 'uint32') {
      new Uint32Array(arrayBuffer, offset, d.data.length).set(d.data);
    } else {
      new Float32Array(arrayBuffer, offset, d.data.length).set(d.data);
    }
  });

  console.log(currentOffset);
  /*
  TODO: replace currentOffset*4 to currentOffset. Binding size (16) is smaller
  than the minimum binding size (64).
   */
  const uniformBuffer = acquireBuffer(
      device, currentOffset * 4,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
  device.queue.writeBuffer(uniformBuffer, 0, arrayBuffer, 0, currentOffset);

  // const uniformInfo = {
  //   size: currentOffset,
  //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  //   buffer: uniformBuffer
  //  };
  // this.uniformPendingDisposal.push(uniformInfo);

  return {offset: 0, size: currentOffset, buffer: uniformBuffer};
}

function makeUniformsDataView(device, sizeA, sizeB) {
  let uniformsWithType = [{type: 'float32', data: [NaN]}];
  uniformsWithType.push({type: 'int32', data: [sizeA, sizeB]});
  return makeUniforms(device, uniformsWithType);
  /*
  const uniformsDataView = computePadding(uniformsWithType);

  const dimensionsBuffer = device.createBuffer({
    size: uniformsDataView.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
  });
  device.queue.writeBuffer(dimensionsBuffer, 0, uniformsDataView);

  return {
    offset: 0,
    size: uniformsDataView.byteLength,
    buffer: dimensionsBuffer
  };
  */
}


(async () => {
  if (!navigator.gpu) {
    console.log(
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // First Matrix

  const firstMatrix = new Float32Array([1, 2, 3, 4, 5, 6, 7, NaN]);

  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();
  // TODO(memoryleak): below result in Destroyed buffer used in a submit.
  // gpuBufferFirstMatrix.destroy();

  // Second Matrix

  const secondMatrix = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();
  // TODO(memoryleak): below result in Destroyed buffer used in a submit.
  // gpuBufferSecondMatrix.destroy();

  // Result Matrix
  const sizeA = 2;
  const sizeB = 4;
  const resultMatrixBufferSize =
      Float32Array.BYTES_PER_ELEMENT * (sizeA * sizeB);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });


  const uniformBuffer = makeUniformsDataView(device, sizeA, sizeB);

  // Compute shader code (GLSL)

  // Pipeline setup
  const shaderWgsl = getComputeShaderCodeWGSL();
  const computePipeline = device.createComputePipeline({
    // layout: device.createPipelineLayout({bindGroupLayouts:
    // [bindGroupLayout]}),
    compute: {
      module: device.createShaderModule({code: shaderWgsl}),
      entryPoint: 'main'
    }
  });

  const bindGroupLayout = computePipeline.getBindGroupLayout(0);
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: gpuBufferFirstMatrix}},
      {binding: 1, resource: {buffer: gpuBufferSecondMatrix}},
      {binding: 2, resource: {buffer: resultMatrixBuffer}},
      {binding: 3, resource: {buffer: uniformBuffer.buffer}}
    ]
  });
  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(sizeA * sizeB /* x */, 1 /* y */);
  passEncoder.end();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
      resultMatrixBuffer /* source buffer */, 0 /* source offset */,
      gpuReadBuffer /* destination buffer */, 0 /* destination offset */,
      resultMatrixBufferSize /* size */
  );

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);

  // TODO(memoryleak): below is successful.
  resultMatrixBuffer.destroy();

  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  console.log(new Float32Array(arrayBuffer));
})();
