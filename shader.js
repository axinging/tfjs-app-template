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

float binaryOperation(float a, float b) {
  // if((a < 0.0) && (floor(b) < b)){
  //   return NAN;
  // }
  if (b == 0.0) {
    return 1.0;
  }

  return a+b;
}

void main() {
  int index = int(gl_GlobalInvocationID.x);
  resultMatrix.numbers[index] = binaryOperation(firstMatrix.numbers[index], secondMatrix.numbers[index]);
}
`;
}

export function getComputeShaderCodeWGSL(workGroupSize) { 
    return `
      [[block]] struct Uniforms { NAN : u32; xShape : vec4<u32>; wShape : vec4<u32>; outShape : vec4<u32>;};
  
      [[block]] struct Matrix {
        numbers: array<f32>;
      };
  
      [[group(0), binding(0)]] var<storage, read> firstMatrix : Matrix;
      [[group(0), binding(1)]] var<storage, read> secondMatrix : Matrix;
      [[group(0), binding(2)]] var<storage, write> resultMatrix : Matrix;
      [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;
  
      fn vec4BoolToVec4F32(value : vec4<bool>) -> vec4<f32> {
        var res = vec4<f32>(0.0);
        for (var i = 0u; i < 4u; i = i + 1u) {
          if (value[i]) {
            res[i] = 1.0;
          }
        }
        return res;
      }
      fn boolToF32(value : bool) -> f32 {
        if (value) {
          return 1.0;
        }
        return 0.0;
      }
  
      fn binaryOperationVec4(a : vec4<f32>, b : vec4<f32>) -> vec4<f32> {
        return vec4BoolToVec4F32(a >= b);
      }
  
      // Return 0 when NaN
      fn binaryOperation(a : f32, b : f32) -> f32 {
        return a + b;
      }
  
      fn caseF32(index : u32) {
        var result : f32 = firstMatrix.numbers[index] + secondMatrix.numbers[index];
        // Results: [1, 1, 0, 0, 0, 1, 1, 0]
        resultMatrix.numbers[index] = binaryOperation(firstMatrix.numbers[index], secondMatrix.numbers[index]);
      }
  
      fn caseVec4(index : u32) {
        let aVec4 = vec4<f32>(firstMatrix.numbers[0], firstMatrix.numbers[1], firstMatrix.numbers[2], firstMatrix.numbers[3]);
        let bVec4 = vec4<f32>(secondMatrix.numbers[0], secondMatrix.numbers[1], secondMatrix.numbers[2], secondMatrix.numbers[3]);
        let resultVec4 = binaryOperationVec4(aVec4, bVec4);
        // Results: [1, 1, 0, 1, 0, 0, 0, 0]
        resultMatrix.numbers[0] = resultVec4[0];
        resultMatrix.numbers[1] = resultVec4[1];
        resultMatrix.numbers[2] = resultVec4[2];
        resultMatrix.numbers[3] = resultVec4[3];
      }
  
      fn activation(a : f32) -> f32 {
        return a;
      }
  
      fn mm_write(globalId : vec3<u32>, valueIn : f32) {
        var value = valueIn;
        let index : u32 = globalId.x;
        // value = activation(valueIn);
        resultMatrix.numbers[index] = value;
      }
  
      [[stage(compute), workgroup_size(${workGroupSize[0]}, ${workGroupSize[1]}, ${workGroupSize[2]})]]
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let index : u32 = globalId.x;
        let result = firstMatrix.numbers[index] + secondMatrix.numbers[index];
        mm_write(globalId, result);
      }
  `;
    }
  
