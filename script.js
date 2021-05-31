// https://developers.google.com/web/updates/2019/08/get-started-with-gpu-compute-on-the-web
import glslangInit from 'https://unpkg.com/@webgpu/glslang@0.0.8/dist/web-devel/glslang.js';
// import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {computeShaderCode} from './shader.js';

function acquireBuffer(device, byteSize, usage) {
  const newBuffer = device.createBuffer({size: byteSize, usage: usage});
  return newBuffer;
}

function arrayToDataView(arrays, length) {
  const BYTES_PER_ELEMENT = 4;
  const uniformDataView =
      new DataView(new ArrayBuffer(length * BYTES_PER_ELEMENT));

  let dataViewIndex = 0;
  arrays.forEach(array => {
    const arrayData = array.data;

    if (array.type !== 'int32' && array.type !== 'float32') {
      throw new Error(`${array.type} not supported!`);
    }

    if (array.type === 'int32') {
      arrayData.forEach(d => {
        uniformDataView.setInt32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
        dataViewIndex++;
      });
    } else {
      arrayData.forEach(d => {
        uniformDataView.setFloat32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
        dataViewIndex++;
      });
    }
  });

  return uniformDataView;
}

function computePadding(uniformsWithType) {
  let currentOffset = 0;
  let padding = 0;
  let dataViewIndex = 0;
  const dimUniformsData = [];
  uniformsWithType.forEach((d, i) => {
    if (d.data.length === 0) {
      d.data = [1];
    }
    // Complete std140 layout rules are documented here:
    // tslint:disable-next-line:max-line-length
    // https://www.khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159
    let baseAlignment;
    switch (d.data.length) {
      case 0:
        baseAlignment = 1;
        break;
      case 1:
        baseAlignment = 1;
        break;
      case 2:
        baseAlignment = 2;
        break;
      case 3:
        baseAlignment = 4;
        break;
      case 4:
        baseAlignment = 4;
        break;
      default:
        util.assert(false, () => `Unsupported ${d.data.length}D shape`);
    }

    padding = Math.ceil(currentOffset / baseAlignment) * baseAlignment -
        currentOffset;
    for (let p = 0; p < padding; ++p) {
      dimUniformsData.push({type: 'int32', data: [0]});
      dataViewIndex++;
    }
    dimUniformsData.push({type: d.type, data: d.data});
    dataViewIndex = dataViewIndex + d.data.length;
    currentOffset += d.data.length + padding;
  });

  return arrayToDataView(dimUniformsData, dataViewIndex);
}


function makeUniformsDataView(device, sizeA, sizeB) {
  let uniformsWithType = [{type: 'float32', data: [NaN]}];
  uniformsWithType.push({type: 'int32', data: [sizeA, sizeB]});
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

  // Bind group layout and bind group

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: 'read-only-storage'}
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: 'read-only-storage'}
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: 'storage'}
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: 'uniform'}
      }
    ]
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: gpuBufferFirstMatrix}},
      {binding: 1, resource: {buffer: gpuBufferSecondMatrix}},
      {binding: 2, resource: {buffer: resultMatrixBuffer}},
      {binding: 3, resource: {buffer: uniformBuffer.buffer}}
    ]
  });

  // Compute shader code (GLSL)

  // Pipeline setup

  const glslang = await glslangInit();

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
    computeStage: {
      module: device.createShaderModule(
          {code: glslang.compileGLSL(computeShaderCode, 'compute')}),
      entryPoint: 'main'
    }
  });

  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(sizeA * sizeB /* x */, 1 /* y */);
  passEncoder.endPass();

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
