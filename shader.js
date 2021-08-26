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
   
      fn binaryOperation(a : f32, b : f32) -> f32 {
        return a + b;
      }
  
      [[stage(compute), workgroup_size(${workGroupSize[0]}, ${workGroupSize[1]}, ${workGroupSize[2]})]]
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let index : u32 = globalId.x;
        resultMatrix.numbers[index] = binaryOperation(firstMatrix.numbers[index], secondMatrix.numbers[index]);
      }
  `;
    }
  
