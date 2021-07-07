// https://developers.google.com/web/updates/2019/08/get-started-with-gpu-compute-on-the-web
import glslangInit from 'https://unpkg.com/@webgpu/glslang@0.0.8/dist/web-devel/glslang.js';

import {computeShaderCodeGLSL} from './conv2d_shader_glsl.js';
// import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
import {computeShaderCodeWGSL} from './conv2d_shader_wgsl.js';

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
        uniformDataView.setFloat32(
            dataViewIndex * BYTES_PER_ELEMENT, d, true);
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

/*
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
*/

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

function computeStrides(shape) {
  const rank = shape.length;
  if (rank < 2) {
    return [];
  }

  // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
  // strides.
  const strides = new Array(rank - 1);
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

function getInputs() {
  const inputDepth = 3;
  const xSize = 8;
  const inputShape = [1, xSize, xSize, inputDepth];
  const outputDepth = 64;
  const fSize = 3;
  const pad = 'valid';
  const stride = [2, 2];

  const inputData = [];
  for (let i = 0; i < xSize * xSize * inputDepth; i++) {
    inputData.push(i % 5);
  }

  const wData = [];
  for (let i = 0; i < fSize * fSize * inputDepth * outputDepth; i++) {
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

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // From tfjsconv2d_webgpu_test 'conv2d x=[1,8,8,3] f=[3,3,3,64] s=[2,2] d=1
  // p=valid Conv2DMMVec4Program remainder != 0'. First Matrix

  const [firstMatrix, secondMatrix] = getInputs();

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

  const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (3 * 3 * 64);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const useWGSL = true;
  // let uniformsWithType: Array<{type: string; data: number[];}> =
  const outputShape = [1, 3, 3, 64];

  let uniformsWithType = [{type: 'float32', data: [NaN]}];
  const bufferShapes = [[1, 8, 8, 3], [3, 3, 3, 64], [1, 3, 3, 64]];
  let uniformsType = 'int32';
  if (useWGSL) {
    uniformsType = 'uint32';
  }
  bufferShapes.map(d => {
    uniformsWithType.push({type: uniformsType, data: d});
  });
  const strides = computeStrides(outputShape);
  // strides 0: 576   1: 192  2: 64
  uniformsWithType.push({type: uniformsType, data: strides});
  // if (program.size != null) {
  //  uniformsWithType.push({type: uniformsType, data: [program.size]});
  //}

  const programUniforms = [
    {
      type: 'int32',
      data: [3, 3]
    },  //[convInfo.filterHeight, convInfo.filterWidth]},
    {type: 'int32', data: [0, 0]},  //[...padInfo]},
    {
      type: 'int32',
      data: [1, 1]
    },  //[convInfo.strideHeight, convInfo.strideWidth]},
    {
      type: 'int32',
      data: [2, 2]
    },  //[convInfo.dilationHeight, convInfo.dilationWidth]}
  ];
  if (useWGSL)
  programUniforms.push(
    {type: 'int32', data: [9]}, //dimAOuter
    {type: 'int32', data: [64]}, //dimBOuter
    {type: 'int32', data: [27]}); //dimInner

  if (programUniforms) {
    uniformsWithType = [...uniformsWithType, ...programUniforms];
  }

  let uniforms = null;
  const uniformsDataView = computePadding(uniformsWithType);
  const uniformsByteLength = uniformsDataView.byteLength;
  const uniformBuffer = makeUniformsDataView(device, uniformsDataView);


  // const uniformBuffer = makeUniformsDataView2(device, sizeA, sizeB);

  // Bind group layout and bind group
  /*
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
  */

  // Compute shader code (GLSL)

  // Pipeline setup

  const glslang = await glslangInit();


  let computePipeline;
  if (useWGSL) {
    computePipeline = device.createComputePipeline({
      // layout: device.createPipelineLayout({bindGroupLayouts:
      // [bindGroupLayout]}),
      computeStage: {
        module: device.createShaderModule({code: computeShaderCodeWGSL}),
        entryPoint: 'main'
      }
    });
  } else {
    computePipeline = device.createComputePipeline({
      // layout: device.createPipelineLayout({bindGroupLayouts:
      // [bindGroupLayout]}),
      computeStage: {
        module: device.createShaderModule(
            {code: glslang.compileGLSL(computeShaderCodeGLSL, 'compute')}),
        entryPoint: 'main'
      }
    });
  }
  const bindGroupLayout = computePipeline.getBindGroupLayout(0);
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {binding: 0, resource: {buffer: resultMatrixBuffer}},
      {binding: 1, resource: {buffer: gpuBufferFirstMatrix}},
      {binding: 2, resource: {buffer: gpuBufferSecondMatrix}},
      {binding: 3, resource: {buffer: uniformBuffer.buffer}}
    ]
  });
  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(1 /* x */, 1 /* y */, 1);
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
