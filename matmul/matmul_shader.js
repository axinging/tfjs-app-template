// https://cnugteren.github.io/tutorial/pages/page3.html
export function getComputeShaderCodeWGSL(workGroupSize) {
  return `
  [[block]] struct Uniforms { NAN : u32; aShape : vec4<u32>; bShape : vec4<u32>; outShape : vec4<u32>;};

  [[block]] struct Matrix0 {
    numbers: array<f32>;
  };

  [[block]] struct Matrix1 {
    numbers: array<f32>;
  };
  [[group(0), binding(0)]] var<storage, read> A : Matrix1;
  
  [[block]] struct Matrix2 {
    numbers: array<f32>;
  };
  [[group(0), binding(1)]] var<storage, read> B : Matrix2;  
  [[group(0), binding(2)]] var<storage, write> result : Matrix0;
  [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;

  [[stage(compute), workgroup_size(${workGroupSize[0]}, ${workGroupSize[1]}, ${workGroupSize[2]})]]

  fn main([[builtin(local_invocation_id)]] localId : vec3<u32>,
        [[builtin(global_invocation_id)]] globalId : vec3<u32>) {
    let M = uniforms.aShape.y;
    let N = uniforms.bShape.y;
    let K = uniforms.aShape.z;

    // Thread identifiers
    let globalRow = globalId.x;
    let globalCol = globalId.y;

    // Compute a single element (loop over K)
    var acc = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
        acc = acc + A.numbers[k*M + globalRow] * B.numbers[globalCol*K + k];
    }
    // Store the result
    result.numbers[globalCol*M + globalRow] = acc;
  }`;
}

// https://cnugteren.github.io/tutorial/pages/page3.html
export function getComputeShaderCodeGLSL(workGroupSize) {
  return `
#version 450
  layout (local_size_x = ${workGroupSize[0]},
            local_size_y = ${workGroupSize[1]},
            local_size_z = ${workGroupSize[2]}) in;
  
  layout(std430, set = 0, binding = 0) readonly buffer ssbx {
    float A[];
  };

  layout(std430, set = 0, binding = 1) readonly buffer ssbW {
    float B[];
  };

  layout(std430, set = 0, binding = 2) writeonly buffer ssbOut {
    float result[];
  };

  layout(std140, set = 0, binding = 3) uniform Uniforms {
    float NAN; ivec4 aShape; ivec4 bShape; ivec4 outShape;
  };
  
  void setOutput(int d0, int d1, int d2, int d3, float value) {
    result[0] = value;
  }
 
  void main() {
    uint M = aShape.y, N = bShape.y, K = aShape.z;

    // Thread identifiers
    uint globalRow = gl_GlobalInvocationID.x;
    uint globalCol = gl_GlobalInvocationID.y;

    // Compute a single element (loop over K)
    float acc = 0.0;
    for (uint k=0u; k<K; k++)
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    // Store the result
    result[globalCol*M + globalRow] = acc;
  }`;
}

