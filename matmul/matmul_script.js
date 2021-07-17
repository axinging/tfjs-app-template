// https://developers.google.com/web/updates/2019/08/get-started-with-gpu-compute-on-the-web
import glslangInit from 'https://unpkg.com/@webgpu/glslang@0.0.8/dist/web-devel/glslang.js';

import {getComputeShaderCodeGLSL2} from './matmul2_shader_glsl.js';
import {getComputeShaderCodeGLSL} from './matmul_shader_glsl.js';
import {getComputeShaderCodeWGSL} from './matmul_shader_wgsl.js';

const useAutoLayout = false;
// when useAutoLayout is true, complains: numBindings mismatch

function getURLState(url) {
  let params = new URLSearchParams(url);
  const keys = [...params.keys()];
  if (keys.length === 0) return 0;
  if (params.has('case')) {
    return Number(params.get('case'));
  }
  return 0;
}

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

    if (array.type !== 'int32' && array.type !== 'float32' &&
        array.type !== 'uint32') {
      throw new Error(`${array.type} not supported!`);
    }

    if (array.type === 'int32') {
      arrayData.forEach(d => {
        uniformDataView.setInt32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
        dataViewIndex++;
      });
    } else if (array.type === 'uint32') {
      arrayData.forEach(d => {
        uniformDataView.setUint32(dataViewIndex * BYTES_PER_ELEMENT, d, true);
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

function makeUniformsDataView(device, uniformsDataView) {
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

function getInputs(M, K, N) {
  const inputData = [];
  for (let i = 0; i < M * K; i++) {
    inputData.push(i % 5);
  }

  const wData = [];
  for (let i = 0; i < K * N; i++) {
    wData.push(i % 5);
  }
  return [new Float32Array(inputData), new Float32Array(wData)];
}


(async () => {
  if (!navigator.gpu) {
    console.log(
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }
  let useWGSL = false;
  const algoSelector = getURLState(window.location.search);
  let getComputeShaderCode = getComputeShaderCodeGLSL;
  if (algoSelector == 2) {
    getComputeShaderCode = getComputeShaderCodeGLSL2;
  } else if (algoSelector == 100) {
    getComputeShaderCode = getComputeShaderCodeWGSL;
  }
  if (algoSelector >= 100) {
    useWGSL = true;
  }
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const M = 64, N = M, K = M;

  const [firstMatrix, secondMatrix] = getInputs(M, K, N);

  // First matrix.
  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix.
  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();

  // Result Matrix.
  const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (M * N);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  const outputShape = [1, M, N, 1];

  let uniformsWithType = [{type: 'float32', data: [NaN]}];
  const bufferShapes = [[1, M, K, 1], [1, K, N, 1], [1, M, N, 1]];
  let uniformsType = 'int32';
  if (useWGSL) {
    uniformsType = 'uint32';
  }
  bufferShapes.map(d => {
    uniformsWithType.push({type: uniformsType, data: d});
  });

  let uniforms = null;
  const uniformsDataView = computePadding(uniformsWithType);
  const uniformsByteLength = uniformsDataView.byteLength;
  const uniformBuffer = makeUniformsDataView(device, uniformsDataView);

  // Bind group layout and bind group
  let bindGroupLayout;
  if (useAutoLayout == false) {
    bindGroupLayout = device.createBindGroupLayout({
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
  }

  const glslang = await glslangInit();
  let computePipeline;
  const workgroupSize = [16, 16, 1];

  let module;
  if (useWGSL) {
    module =
        device.createShaderModule({code: getComputeShaderCode(workgroupSize)});
  } else {
    module = device.createShaderModule({
      code: glslang.compileGLSL(getComputeShaderCode(workgroupSize), 'compute')
    })
  }

  if (useAutoLayout) {
    computePipeline = device.createComputePipeline(
        {computeStage: {module: module, entryPoint: 'main'}});
    bindGroupLayout = computePipeline.getBindGroupLayout(0);
  } else {
    computePipeline = device.createComputePipeline({
      layout:
          device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
      computeStage: {module: module, entryPoint: 'main'}
    });
  }
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
  passEncoder.dispatch(
      M / workgroupSize[0] /* x */, N / workgroupSize[1] /* y */, 1);
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

  gpuBufferFirstMatrix.destroy();
  gpuBufferSecondMatrix.destroy();
  resultMatrixBuffer.destroy();

  // Read buffer.
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  console.log(new Float32Array(arrayBuffer));
})();
