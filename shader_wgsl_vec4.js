export function getComputeShaderCodeWGSL(workGroupSize = [4, 1, 1]) {
  return `
    struct Uniforms { NAN : u32, xShape : vec2<u32>};

    struct Matrix {
      numbers: array<vec4<f32>>,
    };

    @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
    @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
    @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;
    @group(0) @binding(3) var<uniform> uniforms : Uniforms;

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
      // pass: {"0":2,"1":4,"2":6,"3":8,"4":10,"5":12,"6":14,"7":null}
      /*
      for (var i = 0; i < 4; i = i + 1) {
        clampedValue[i] = result[i];
      }
      */

      // fail: {"0":2,"1":4,"2":6,"3":8,"4":null,"5":null,"6":null,"7":null}
      for (var i = 0; i < 4; i = i + 1) {
        if (isnan(result[i])) {
          clampedValue[i] = result[i];
        } else {
          clampedValue[i] = clamp(result[i], 0.0, 10.0);
        }
      }
      // pass {"0":10,"1":10,"2":10,"3":10,"4":10,"5":10,"6":10,"7":null}
      /*
      for (var i = 0; i < 4; i = i + 1) {
        if (isnan(result[i])) {
          clampedValue[i] = result[i];
        } else {
          clampedValue[i] = 10;// clamp(result[i], 0.0, 10.0);
        }
      }
      */
      
      // pass: {"0":2,"1":4,"2":6,"3":8,"4":10,"5":12,"6":14,"7":null}
      /*
      var i = 0;
      if (isnan(result[i])) {
        clampedValue[i] = result[i];
      } else {
        clampedValue[i] = clamp(result[i], 0.0, 10.0);
      }
      i = 1;
      if (isnan(result[i])) {
        clampedValue[i] = result[i];
      } else {
        clampedValue[i] = clamp(result[i], 0.0, 10.0);
      }
      i = 2;
      if (isnan(result[i])) {
        clampedValue[i] = result[i];
      } else {
        clampedValue[i] = clamp(result[i], 0.0, 10.0);
      }
      i = 3;
      if (isnan(result[i])) {
        clampedValue[i] = result[i];
      } else {
        clampedValue[i] = clamp(result[i], 0.0, 10.0);
      }
      */

      /*
      if (isnan(result.r)) {
        clampedValue.r = result.r;
      } else {
        clampedValue.r = clamp(result.r, 0.0, 10.0);
      }
      if (isnan(result.g)) {
        clampedValue.g = result.g;
      } else {
        clampedValue.g = clamp(result.g, 0.0, 10.0);
      }

      if (isnan(result.b)) {
        clampedValue.b = result.b;
      } else {
        clampedValue.b = clamp(result.b, 0.0, 10.0);
      }
      if (isnan(result.a)) {
        clampedValue.a = result.a;
      } else {
        clampedValue.a = clamp(result.a, 0.0, 10.0);
      }
      */
      // OK 1
      /*
      if (isnan(result.r)) {
        clampedValue.r = result.r;
      } else {
        clampedValue.r = clamp(result.r, 0.0, 10.0);
      }
      if (isnan(result.g)) {
        clampedValue.g = result.g;
      } else {
        clampedValue.g = clamp(result.g, 0.0, 10.0);
      }

      if (isnan(result.b)) {
        clampedValue.b = result.b;
      } else {
        clampedValue.b = clamp(result.b, 0.0, 10.0);
      }
      if (isnan(result.a)) {
        clampedValue.a = result.a;
      } else {
        clampedValue.a = clamp(result.a, 0.0, 10.0);
      }*/
      setOutputAtIndex(i32(index), clampedValue);
    }
`;
}
