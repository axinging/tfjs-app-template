export function getComputeShaderCodeWGSL(workGroupSize = [4, 1, 1]) {
  return `
    struct Uniforms { NAN : u32, xShape : vec4<u32>, wShape : vec4<u32>, outShape : vec4<u32>,};

    struct Matrix {
      numbers: array<vec4<f32>>,
    };

    @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
    @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
    @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;
    @group(0) @binding(3) var<uniform> uniforms : Uniforms;

  
    fn activation(a : f32) -> f32 {
      return a;
    }


    fn setOutputAtIndex(flatIndex : i32, value : vec4<f32>) {
      resultMatrix.numbers[flatIndex] = vec4<f32>(value);
    }

    fn isnan(val: f32) -> bool {
      let floatToUint: u32 = bitcast<u32>(val);
      return (floatToUint & 0x7fffffffu) > 0x7f800000u;
    }

    @compute @workgroup_size(${workGroupSize[0]}, ${
      workGroupSize[1]}, ${workGroupSize[2]})
    fn main(@builtin(global_invocation_id) globalId : vec3<u32>) {
      let index : u32 = globalId.x;
      let result = firstMatrix.numbers[index] + secondMatrix.numbers[index];
      if (u32(result.r) == uniforms.NAN) {

      }
      var clampedValue : vec4<f32>;
      for (var i = 0; i < 4; i = i + 1) {
        if (isnan(result[i])) {
          clampedValue[i] = result[i];
        } else {
          clampedValue[i] = clamp(result[i], 0.0, 10.0);
        }
      }
      setOutputAtIndex(i32(index), clampedValue);
    }
`;
}
