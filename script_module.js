const useAutoLayout = true;

let langOption = 'wgsl';
let caseOption = 0;

function getURLState(url) {
  let params = new URLSearchParams(url);
  const keys = [...params.keys()];
  if (keys.length === 0) return false;
  if (params.has('case')) {
    caseOption = Number(params.get('case'));
  }
  if (params.has('lang')) {
    langOption = params.get('lang');
  }
  return langOption === 'wgsl';
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

async function executeMatmul(device, firstMatrix, secondMatrix, size, useWGSL) {
  var wgslFuncs = {
    0: getComputeShaderCodeWGSL,
  };
  var getComputeShaderCode;
  if (useWGSL) {
    getComputeShaderCode = wgslFuncs[caseOption];
  } 
  // First matrix.
  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix
  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();
  const workgroupSize = [size, 1, 1];

  // Result Matrix
  const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (size);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  const outputShape = [1, 1, 1, size];

  let uniformsWithType = [{type: 'float32', data: [NaN]}];
  const bufferShapes = [[1, 1, 1, size], [1, 1, 1, size], [1, 1, 1, size]];
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
  let computeCode = getComputeShaderCode(workgroupSize);
  let module = device.createShaderModule(
      {code: await getModuleCode(computeCode, useWGSL)}, 'compute');

  const [computePipeline, bindGroupLayout] =
      getComputePipeline(device, module, useWGSL);

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
  passEncoder.dispatch(size /* x */, 1 /* y */);
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
  return gpuReadBuffer.getMappedRange();
}

async function getDevice() {
  if (!navigator.gpu) {
    console.log(
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  return await adapter.requestDevice();
}

(async () => {
  const device = await getDevice();

  // First Matrix
  const size = 8;
  const firstMatrix = new Float32Array([1, 2, 3, NaN, 4, 5, 7, NaN]);

  const secondMatrix = new Float32Array([0, 2, 4, NaN, NaN, 5, 6, NaN]);

  let useWGSL = true;//getURLState(window.location.search);
  {
    const arrayBuffer =
        await executeMatmul(device, firstMatrix, secondMatrix, size, useWGSL);
    console.log(new Float32Array(arrayBuffer));
  }
  //{
  //  const arrayBuffer =
  //      await executeMatmul(device, firstMatrix, secondMatrix, size, true);
  //  console.log(new Float32Array(arrayBuffer));
  //}
})();

async function getModuleCode(computeCode, useWGSL) {
  if (useWGSL) {
    return computeCode;
  }
  const glslang = await glslangInit();
  return glslang.compileGLSL(computeCode, 'compute');
}

function getComputePipeline(device, module, useWGSL) {
  let computePipeline;
  let bindGroupLayout;
  const pipelineConstant = {
    0: false,  // "has_point_light"
    1: 300.0,  // "specular_param"
  };
  computePipeline = device.createComputePipeline({
    compute:
        {module: module, entryPoint: 'main', constants: pipelineConstant}
  });
  bindGroupLayout = computePipeline.getBindGroupLayout(0);
  return [computePipeline, bindGroupLayout];
}


export function getComputeShaderCodeWGSL(workGroupSize) {
  return `
      // [[override(0)]] let has_point_light: bool = true; // Algorithmic control.
      // [[override(1)]] let specular_param: f32 = 2.3;    // Numeric control.
      [[block]] struct Uniforms { NAN : u32; xShape : vec4<u32>; wShape : vec4<u32>; outShape : vec4<u32>;};

      [[block]] struct Matrix {
        numbers: array<f32>;
      };

      [[group(0), binding(0)]] var<storage, read> firstMatrix : Matrix;
      [[group(0), binding(1)]] var<storage, read> secondMatrix : Matrix;
      [[group(0), binding(2)]] var<storage, write> resultMatrix : Matrix;
      [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;

      fn binaryOperation(a : f32, b : f32) -> f32 {
        return a + b;
      }

      // If uniforms is not used, will complain: Number of entries (4) did not match the number of entries (3) specified in [BindGroupLayout]
      fn binaryOperationPow(a : f32, b : f32) -> f32 {
        if(a < 0.0 && floor(b) < b) {
          return f32(uniforms.NAN);
        }
        if (b == 0.0) {
          return 1.0;
        }
        if (i32(round(b % 2.0)) != 1) {
          return pow(abs(a), b);
        }
        return sign(a) * pow(abs(a), b);
      }

      [[stage(compute), workgroup_size(${workGroupSize[0]}, ${
      workGroupSize[1]}, ${workGroupSize[2]})]]
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let index : u32 = globalId.x;
        resultMatrix.numbers[index] = binaryOperationPow(firstMatrix.numbers[index], secondMatrix.numbers[index]);// + specular_param;
      }
  `;
}
