export const computeShaderCode = `


    fn idiv(a: i32, b: i32, sign: f32) -> i32 {
      var res: i32 = a / b;
      let mod: i32 = a % b;
      if (sign < 0. && mod != 0) {
        res = res - 1;
      }
      return res;
    }
    
    fn greaterThanEqual(coord: vec4<i32>, data: vec4<i32>) -> vec4<bool> {
        return vec4<bool> (coord >= data);
    }
    fn greaterThanEqualv3(coord: vec3<i32>, data: vec3<i32>)  -> vec3<bool>{
      return vec3<bool> (coord >= data);
    }
    fn greaterThanEqualv2(coord: vec2<i32>, data: vec2<i32>)  -> vec2<bool>{
      return vec2<bool> (coord >= data);
    }
    // GREATER_THAN_EQUAL

    // Checks whether coordinates lie within the bounds of the shape.
    fn coordsInBounds4(coord: vec4<i32>, shape: vec4<i32>) -> bool {
      return all(coord >= vec4<i32>(0, 0, 0, 0)) &&
          all(coord < shape);
    }

    fn coordsInBounds3(coord: vec3<i32>, shape: vec3<i32>) -> bool{
      return all(coord >= vec3<i32>(0, 0, 0)) &&
          all(coord < shape);
    }

    fn coordsInBounds2(coord: vec2<i32>, shape: vec2<i32>) -> bool {
      return all(coord >= vec2<i32>(0, 0)) &&
          all(coord < shape);
    }

    // float NAN; int sizeA; int sizeB;
    [[block]] struct Uniforms {
      NAN : f32;
      size: vec2<u32>;
    };

    [[block]] struct Matrix {
      numbers: array<f32>;
    };

    [[group(0), binding(0)]] var<storage> firstMatrix : [[access(read)]] Matrix;
    [[group(0), binding(1)]] var<storage> secondMatrix : [[access(read)]] Matrix;
    [[group(0), binding(2)]] var<storage> resultMatrix : [[access(write)]] Matrix;
    [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;

    // let sizeA : u32 = uniforms.size[0]; // This not work!

    [[stage(compute)]] fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
      let index : u32 = global_id.x;
      var result : f32 = firstMatrix.numbers[index] + secondMatrix.numbers[index];
      let sizeA : u32 = uniforms.size[0];
      resultMatrix.numbers[index] = f32(sizeA); //f32(uniforms.size[0]);
    }
`;
