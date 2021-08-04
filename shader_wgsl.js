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
      return boolToF32(a >= b);
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

    [[stage(compute), workgroup_size(${workGroupSize[0]}, ${workGroupSize[1]}, ${workGroupSize[2]})]]
    fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
      let index : u32 = global_id.x;
      // First: [1, 2, 3, NaN, 4, 5, 7, NaN]
      // Second: [0, 2, 4, NaN, NaN, 5, 6, NaN]
      // caseF32(index);
      caseVec4(index);
    }
`;
  }
