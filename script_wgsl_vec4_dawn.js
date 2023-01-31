// https://developers.google.com/web/updates/2019/08/get-started-with-gpu-compute-on-the-web
// import glslangInit from '@webgpu/glslang/dist/web-devel/glslang.onefile';
function acquireBuffer(device, byteSize, usage) {
  const newBuffer = device.createBuffer({size: byteSize, usage: usage});
  return newBuffer;
}

function getComputeShaderCodeWGSL() {
    return ` 
      struct Buf {
        data: array<vec4<f32>, 2>,
      };
  
      @group(0) @binding(0) var<storage, read> inBuf : Buf;
      @group(0) @binding(1) var<storage, read_write> outBuf : Buf; 
 
      fn isnan(val: f32) -> bool {
        let floatToUint: u32 = bitcast<u32>(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }
  
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) globalId : vec3<u32>) {
        
          for (var j = 0; j < 2; j ++) {
              var clampedValue : vec4<f32>;
              let result = inBuf.data[j];
              for (var i = 0; i < 4; i = i + 1) {
                  if (isnan(result[i])) {
                      clampedValue[i] = result[i];
                  } else {
                      clampedValue[i] = clamp(result[i], 0.0, 10.0);
                  }
              }
              outBuf.data[j] = vec4<f32>(clampedValue);
          }
      }
  `;
  }


  function getComputeShaderCodeWGSL1() {
    return ` 
      struct Buf {
        data: array<vec4<f32>, 2>,
      };
  
      @group(0) @binding(0) var<storage, read> inBuf : Buf;
      @group(0) @binding(1) var<storage, read_write> outBuf : Buf; 
 
      fn isnan(val: f32) -> bool {
        let floatToUint: u32 = bitcast<u32>(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }
  
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) globalId : vec3<u32>) {
        
        for (var j = 0; j < 2; j ++) {
            // let result = inBuf.data[j];
            for (var i = 0; i < 4; i = i + 1) {
                if (isnan(inBuf.data[j][i])) {
                  outBuf.data[j][i] = inBuf.data[j][i];
                } else {
                  outBuf.data[j][i] = clamp(inBuf.data[j][i], 0.0, 10.0);
                }
            }
        }
      }
  `;
  }


  /*
          for (var j = 0; j < 2; j ++) {
            // let result = inBuf.data[j];
            for (var i = 0; i < 4; i = i + 1) {
                if (isnan(inBuf.data[j][i])) {
                  outBuf.data[j][i] = inBuf.data[j][i];
                } else {
                  outBuf.data[j][i] = clamp(inBuf.data[j][i], 0.0, 10.0);
                }
            }
        }
  */

(async () => {
  if (!navigator.gpu) {
    console.log(
        'WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.');
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // First Matrix

  const inBuf = new Float32Array([0, 1, 2, 3, 4, 5, 6, NaN]);

  const gpuBufferFirstMatrix = device.createBuffer({
    mappedAtCreation: true,
    size: inBuf.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  const arrayBufferFirstMatrix = gpuBufferFirstMatrix.getMappedRange();
  new Float32Array(arrayBufferFirstMatrix).set(inBuf);
  gpuBufferFirstMatrix.unmap();
  // TODO(memoryleak): below result in Destroyed buffer used in a submit.
  // gpuBufferFirstMatrix.destroy();

  // Result Matrix
  const sizeA = 2;
  const sizeB = 4;
  const resultMatrixBufferSize =
      Float32Array.BYTES_PER_ELEMENT * (sizeA * sizeB);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Compute shader code (GLSL)

  // Pipeline setup
  const shaderWgsl = getComputeShaderCodeWGSL();
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
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
      {binding: 1, resource: {buffer: resultMatrixBuffer}}
    ]
  });
  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const workPerThread = 4;
  passEncoder.dispatchWorkgroups(
      1/* x */, 1 /* y */);
  passEncoder.end();

  // Get a GPU buffer for reading in an unmapped state.
  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  console.log(resultMatrixBufferSize);

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
  console.log(JSON.stringify(new Float32Array(arrayBuffer)));
})();
