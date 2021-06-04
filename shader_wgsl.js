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

    fn getFlatIndex(coord : u32, shape : u32) -> u32 {
      return coord;
    }

    fn getFlatIndex2(coords : vec2<u32>, shape : vec2<u32>) -> u32 {
      return u32(dot(vec2<f32>(coords), vec2<f32>(f32(shape.y), 1.0)));
    }

    fn getFlatIndex3(coords : vec3<u32>, shape : vec3<u32>) -> u32 {
      return u32(dot(vec3<f32>(coords), vec3<f32>(f32(shape.y) * f32(shape.z), f32(shape.z), 1.0)));
    }

    fn getFlatIndex4(coords : vec4<u32>, shape : vec4<u32>) -> u32 {
      return u32(dot(vec4<f32>(coords), vec4<f32>(
          f32(shape.y) * f32(shape.z) * f32(shape.w), f32(shape.z) * f32(shape.w), f32(shape.w), 1.0)));
    }

    fn dottest(a: vec2<f32>, b : vec2<f32>) ->f32 {
      return dot(a, b);
    }

    fn dottestu32(a: vec2<u32>, b : vec2<u32>) ->f32 {
      return dot(a, b);
    }

    fn dottest4(a: vec4<f32>, b : vec4<f32>) ->f32 {
      return dot(a, b);
    }
    // error: cannot assign to value of type 'u32'
    fn inputVar(index: u32) ->u32 {
       index = index - 3u;
       let a : u32 = index;
       return a;
    }

    fn inputVar2(index: u32) ->u32 {
      var index2 : u32 = index - 3u;
      let a : u32 = index2;
      return a;
    }

    fn conditionExpr() -> i32{
      let a : i32 = 2;
      let b : i32 = 3;
      let c : i32 = 0;
      // let d : i32 = c > 0 ? a : b; 
      var d : i32;
      if (c > 0) {
        d = a;
      } else {
        d = b;
      }
      return d;
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

    // let sizeA : u32 = uniforms.size[0]; // Not work ! uniforms can not be used as global.
    // let gidx : vec3<u32> = global_invocation_id;

    // let TileSize : u32 = 4;

    var<workgroup> mm_Asub : array<f32, 4>;
    // wg_size not work!
    // [[builtin(workgroup_size)]] wg_size : vec3<u32>
    // resultMatrix.numbers[index] = f32(wg_size.x);

    [[stage(compute), workgroup_size(16, 16, 1)]]
    fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
      let index : u32 = global_id.x;
      var result : f32 = firstMatrix.numbers[index] + secondMatrix.numbers[index];
      let sizeA : u32 = uniforms.size[0];
      // resultMatrix.numbers[index] = f32(global_id.x); //f32(uniforms.size[0]);
      resultMatrix.numbers[index] = dottest4(vec4<f32>(1.0,1.0,1.0,1.0), vec4<f32>(1.0,1.0,1.0,2.0));

    }
`;
