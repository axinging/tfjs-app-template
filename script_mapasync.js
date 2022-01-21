import glslangInit from 'https://unpkg.com/@webgpu/glslang@0.0.15/dist/web-devel/glslang.js';
// import {getComputeShaderCodeGLSL, getComputeShaderCodeWGSL} from
// './shader.js';
const useAutoLayout = true;

let langOption = 'wgsl';
let caseOption = 0;

function getURLState(url) {
  let params = new URLSearchParams(url);
  const keys = [...params.keys()];
  if (keys.length === 0) return true;
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

function defaultGpuBufferUsage() {
  return GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST;
}

function upload(device, firstMatrix, secondMatrix) {
  // First matrix.
  const start1 = performance.now();
  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix
  const start2 = performance.now();
  const gpuBufferSecondMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();
  const end = performance.now();
  /*
  console.log(
      'uploadWithMap: ' + (start2 - start1).toFixed(3) + ', ' +
      (end - start2).toFixed(3));
  */
  return [
    gpuBufferFirstMatrix, gpuBufferSecondMatrix, (start2 - start1).toFixed(3),
    (end - start2).toFixed(3)
  ];
}


function upload2(device, firstMatrix, secondMatrix) {
  // First matrix.
  const start1 = performance.now();
  const gpuUsage = defaultGpuBufferUsage();
  const gpuBufferFirstMatrix =
      device.createBuffer({size: firstMatrix.byteLength, usage: gpuUsage});

  device.queue.writeBuffer(gpuBufferFirstMatrix, 0, firstMatrix);

  // Second Matrix
  const start2 = performance.now();
  const gpuBufferSecondMatrix =
      device.createBuffer({size: secondMatrix.byteLength, usage: gpuUsage});

  device.queue.writeBuffer(gpuBufferSecondMatrix, 0, secondMatrix);

  const end = performance.now();
  /*
  console.log(
      'uploadWithWriteBuffer: ' + (start2 - start1).toFixed(3) + ', ' +
      (end - start2).toFixed(3));
  */
  return [
    gpuBufferFirstMatrix, gpuBufferSecondMatrix, (start2 - start1).toFixed(3),
    (end - start2).toFixed(3)
  ];
}

function submitQueue(queue, currentCommandEncoder) {
  // this.ensureComputePassEnded();
  queue.submit([currentCommandEncoder.finish()]);
}

const LOOP_SIZE = 1000000000;
function loop(max, tag = '') {
    var sum = 0;
    console.log("loop");
    for (var i = 0; i < max; i++) {
        sum = sum + Math.sqrt(i);
    }
}

async function executeMatmul(device, firstMatrix, secondMatrix, size, useWGSL) {
  // console.time("executeMatmul");
  var glslFuncs = {
    0: getComputeShaderCodeGLSL,
  };
  var wgslFuncs = {
    0: getComputeShaderCodeWGSL,
  };
  var getComputeShaderCode;
  if (useWGSL) {
    getComputeShaderCode = wgslFuncs[caseOption];
  } else {
    getComputeShaderCode = glslFuncs[caseOption];
  }

  const querySet = device.createQuerySet({
    type: 'timestamp',
    count: 2,
  });
  const [gpuBufferFirstMatrix, gpuBufferSecondMatrix] =
      upload(device, firstMatrix, secondMatrix);

  const workgroupSize = [4, 1, 1];

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
  const compile_before = performance.now();
  let module = device.createShaderModule(
      {code: await getModuleCode(computeCode, useWGSL), label: 'ProgramName'},
      'compute');
  // console.log("createShaderModule = " + (performance.now() -
  // compile_before));
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
  var commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(size / 256 /* x */, 256 /* y */);
  passEncoder.endPass();


  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);
  commandEncoder = null;

  gpuBufferFirstMatrix.destroy();
  gpuBufferSecondMatrix.destroy();
  
  // Read buffer.
  await getBufferData(device, resultMatrixBuffer, resultMatrixBufferSize);
  loop(LOOP_SIZE);
  resultMatrixBuffer.destroy();
}



async function getBufferData(
    device, resultMatrixBuffer, resultMatrixBufferSize) {
  console.time('getBufferData');
  // console.log("getBufferData 1");
  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const commandEncoder = device.createCommandEncoder();

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
      resultMatrixBuffer /* source buffer */, 0 /* source offset */,
      gpuReadBuffer /* destination buffer */, 0 /* destination offset */,
      resultMatrixBufferSize /* size */
  );

  // this.ensureCommandEncoderReady();
  // this.ensureComputePassEnded();
  submitQueue(device.queue, commandEncoder);
  console.timeEnd('getBufferData');

  console.time('mapAsync');
  const mapPromise = gpuReadBuffer.mapAsync(GPUMapMode.READ);
  console.timeEnd('mapAsync');
  console.time('awaitMapAsync');
  await mapPromise;
  console.timeEnd('awaitMapAsync');
  const values = gpuReadBuffer.getMappedRange().slice(0);
  console.log(JSON.stringify(values));
  console.log(new Float32Array(values));

  gpuReadBuffer.unmap();

  return values;
}

async function getDevice() {
  if (!navigator.gpu) {
    console.log(
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }

  const gpuDescriptor = {powerPreference: 'high-performance'};

  const adapter = await navigator.gpu.requestAdapter(gpuDescriptor);
  let deviceDescriptor = {};
  const supportTimeQuery = adapter.features.has('timestamp-query');

  if (supportTimeQuery) {
    deviceDescriptor = {requiredFeatures: ['timestamp-query']};
  } else {
    console.warn(
        `This device doesn't support timestamp-query extension. ` +
        `Start Chrome browser with flag ` +
        `--disable-dawn-features=disallow_unsafe_apis then try again. ` +
        `Or zero will shown for the kernel time when profiling mode is` +
        `enabled. Using performance.now is not workable for webgpu since` +
        `it doesn't support synchronously to read data from GPU.`);
  }
  return await adapter.requestDevice(deviceDescriptor);
}

(async () => {
  const device = await getDevice();

  for (var j = 0; j < 1; j++) {
    const batch = 1 * 100000;  // 262145;
    const size = batch * 1;
    // const firstMatrix = new Float32Array(size);
    // const secondMatrix = new Float32Array(size);
    const firstMatrix = new Int32Array(size);
    const secondMatrix = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      firstMatrix[i] = i;
      secondMatrix[i] = i + 10;
    }

    let useWGSL = getURLState(window.location.search);
    {
      const arrayBuffer =
          await executeMatmul(device, firstMatrix, secondMatrix, size, useWGSL);
      console.log(new Float32Array(arrayBuffer));
    }
  }
})();

async function getModuleCode(computeCode, useWGSL) {
  if (useWGSL) {
    return computeCode;
  }
  const glslang = await glslangInit();
  return glslang.compileGLSL(computeCode, 'compute');
}

function getComputePipeline(device, module, useWGSL) {
  // console.time("getComputePipeline");
  let computePipeline;
  let bindGroupLayout;
  const pipelineConstant = {
    0: false,  // "has_point_light"
    1: 300.0,  // "specular_param"
  };
  if (useWGSL) {
    if (useAutoLayout) {
      const compile_before = performance.now();
      computePipeline = device.createComputePipeline({
        compute:
            {module: module, entryPoint: 'main', constants: pipelineConstant}
      });
      // console.log("wgsl createComputePipeline = "+ (performance.now() -
      // compile_before));
      bindGroupLayout = computePipeline.getBindGroupLayout(0);
    } else {
      bindGroupLayout = getBindGroupLayout(device);
      computePipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
        compute:
            {module: module, entryPoint: 'main', constants: pipelineConstant}
      });
    }
  } else {
    if (useAutoLayout) {
      const compile_before = performance.now();
      computePipeline = device.createComputePipeline(
          {compute: {module: module, entryPoint: 'main'}});
      bindGroupLayout = computePipeline.getBindGroupLayout(0);
      // console.log("glsl createComputePipeline = "+ (performance.now() -
      // compile_before));
    } else {
      bindGroupLayout = getBindGroupLayout(device);
      computePipeline = device.createComputePipeline({
        layout:
            device.createPipelineLayout({bindGroupLayouts: [bindGroupLayout]}),
        compute: {module: module, entryPoint: 'main'}
      });
    }
  }
  // console.timeEnd("getComputePipeline");
  return [computePipeline, bindGroupLayout];
}

function getBindGroupLayout(device) {
  return device.createBindGroupLayout({
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

export function getComputeShaderCodeGLSL(workGroupSize) {
  return `#version 450
    layout (local_size_x = ${workGroupSize[0]},
    local_size_y = ${workGroupSize[1]},
    local_size_z = ${workGroupSize[2]}) in;

    layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
      float numbers[];
    } firstMatrix;

    layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
      float numbers[];
    } secondMatrix;

    layout(std430, set = 0, binding = 2) buffer ResultMatrix {
      float numbers[];
    } resultMatrix;

    layout(std140, set = 0, binding = 3) uniform Uniforms {
      float NAN; ivec4 aShape; ivec4 bShape; ivec4 outShape;
    };

    float binaryOperation1(float a, float b) {
      return a + b;
    }

    float binaryOperation(float a, float b) {
      if((a < 0.0) && (floor(b) < b)){
        return NAN;
      }
      if (b == 0.0) {
        return 1.0;
      }
      return (round(mod(b, 2.0)) != 1) ?
        pow(abs(a), b) : sign(a) * pow(abs(a), b);
    }

    void main() {
      int index = int(gl_GlobalInvocationID.x);
      resultMatrix.numbers[index] = binaryOperation(firstMatrix.numbers[index], secondMatrix.numbers[index]);
    }
`;
}
export function getComputeShaderCodeWGSL(workGroupSize) {
  return `
      // [[override(0)]] let has_point_light: bool = true; // Algorithmic control.
      // [[override(1)]] let specular_param: f32 = 2.3;    // Numeric control.
      struct Uniforms { NAN : u32; xShape : vec4<u32>; wShape : vec4<u32>; outShape : vec4<u32>;};

      struct  Matrix {
        numbers: array<f32>;
      };

      [[group(0), binding(0)]] var<storage, read> firstMatrix : Matrix;
      [[group(0), binding(1)]] var<storage, read> secondMatrix : Matrix;
      [[group(0), binding(2)]] var<storage, write> resultMatrix : Matrix;
      [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;
      let a0 = 2147483600;
      let b0 = 47;

      fn dotVec4I32(a : vec4<i32>, b : vec4<i32>) -> i32 {
        /*
        var result = 0;
        for (var i = 0; i < 4; i = i + 1) {
          result = result + a[i]*b[i];
        }
        */
        return dot(a, b);
      }
    
      fn getFlatIndex4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
        return i32(dotVec4I32(coords,
            vec4<i32>(shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1)));
      }

      fn getFlatIndex4D2(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
        return i32(dot(vec4<f32>(coords), vec4<f32>(
          f32(shape.y) * f32(shape.z) * f32(shape.w), f32(shape.z) * f32(shape.w), f32(shape.w), 1.0)));
      }  

      fn binaryOperation_0(a : f32, b : f32) -> f32 {
        //let a = vec4<i32>(1);
        if(a < 0.0 && floor(b) < b) {
          return f32(uniforms.NAN);
        }
        let index = getFlatIndex4D(vec4<i32>(263169,4,4,4),vec4<i32>(263169,4,4,4));
        // return a + b + f32(id);
        //const baseSize = 263169;
        //const size = baseSize * 4 * 4 * 4;
        return f32(a0)+f32(b0);
      }

      fn binaryOperation2(a : f32, b : f32) -> f32 {
        //let a = vec4<i32>(1);
        if(a < 0.0 && floor(b) < b) {
          return f32(uniforms.NAN);
        }
        let index = getFlatIndex4D2(vec4<i32>(263169,4,4,4),vec4<i32>(263169,4,4,4));
        // return a + b + f32(id);
        //const baseSize = 263169;
        //const size = baseSize * 4 * 4 * 4;
        return f32(a0+b0);
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

      fn binaryOperation(a : f32, b : f32) -> f32 {
        return a + b;
      }
      
      [[stage(compute), workgroup_size(${workGroupSize[0]}, ${
      workGroupSize[1]}, ${workGroupSize[2]})]]
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let index : u32 = globalId.x;
        // Avoid uniforms.NAN warning.
        binaryOperationPow(1.0,2.0);
        resultMatrix.numbers[index] = binaryOperation(firstMatrix.numbers[index], secondMatrix.numbers[index]);
      }
  `;
}
