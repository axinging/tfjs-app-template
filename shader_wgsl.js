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
      // dot(vecN<f32>, vecN<f32>) -> f32
      // return dot(a, b);
      return 0.0;

    }

    fn dottest4(a: vec4<f32>, b : vec4<f32>) ->f32 {
      return dot(a, b);
    }
    // bool can not be converted to f32 or i32.
    // error: type in vector constructor does not match vector type: expected 'f32', found 'bool'
    fn unknow() -> vec4<f32> {
      let a : vec4<f32>= vec4<f32>(0.,0., 0., 0.);
      let b : vec4<f32>= vec4<f32>(0.,0., 0., 0.);
      let aLessThanZero : vec4<bool> = vec4<bool>(a < vec4<f32>(0.,0., 0., 0.));
      var aLessThanZeroF32 : vec4<f32> = vec4<f32>(0.,0., 0., 0.); 
      if (aLessThanZero[0]) {
        aLessThanZeroF32[0] = 1.0;
      }
      // var i :u32 = 0u;
      for (var i:u32 = 0u; i< 4u; i = i+1u ) {
        if (aLessThanZero[i]) {
          aLessThanZeroF32[i] = 1.0;
        }
      }
      return (vec4<f32>(aLessThanZeroF32) * (b * a)) + ((vec4<f32>(1.0, 1.0,1.0,1.0) - vec4<f32>(aLessThanZeroF32)) * a);
    }

    // error: cannot assign to value of type 'u32'
    fn inputVar(index: u32) ->u32 {
       //index = index - 3u;
       //let a : u32 = index;
       //return a;
       return 3u;
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

    [[group(0), binding(0)]] var<storage, read> firstMatrix : Matrix;
    [[group(0), binding(1)]] var<storage, read> secondMatrix :  Matrix;
    [[group(0), binding(2)]] var<storage, write> resultMatrix :Matrix;
    [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;

    // let sizeA : u32 = uniforms.size[0]; // Not work ! uniforms can not be used as global.
    // let gidx : vec3<u32> = global_invocation_id;

    // let TileSize : u32 = 4;

    var<workgroup> mm_Asub : array<f32, 4>;
    // wg_size not work!
    // [[builtin(workgroup_size)]] wg_size : vec3<u32>
    // resultMatrix.numbers[index] = f32(wg_size.x);

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
  

    fn lessVec4F32(a : vec4<f32>, b : vec4<f32>) -> vec4<f32> {
      let resultBool = vec4<bool>(a < b);
      return vec4<f32>(boolToF32(resultBool[0]), boolToF32(resultBool[1]),
          boolToF32(resultBool[2]), boolToF32(resultBool[3]));
    }



    fn binaryOperation(a : vec4<f32>, b : vec4<f32>) -> vec4<f32> {
      // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
      // var ptr_vec4f32 = vec4<f32>(79.9);
      // var part_vec4f32 = modf(ptr_vec4f32, &ptr_vec4f32);
      // vec4 isModRound1 = vec4 (round(b % 2.0) == ivec4(1));
      //let isModRound1 = vec4<f32>(vec4<i32>(round(b % vec4<f32>(2.0))) == vec4<i32>(1));
      let isModRound1Bool = vec4<i32>(round(b % vec4<f32>(2.0))) == vec4<i32>(1);
      let isModRound1 = vec4BoolToVec4F32(isModRound1Bool);
      let multiplier = sign(a) * isModRound1 + (vec4<f32>(1.0) - isModRound1);
      var result = multiplier * pow(abs(a), b);

      // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
      let isExpZero = b == vec4<f32>(0.0);
      // result.r = isExpZero.r ? 1.0 : result.r;
      // result.g = isExpZero.g ? 1.0 : result.g;
      // result.b = isExpZero.b ? 1.0 : result.b;
      // result.a = isExpZero.a ? 1.0 : result.a;
      if (isExpZero.r) {
        result.r = 1.0;
      }
      if (isExpZero.g) {
        result.g = 1.0;
      }
      if (isExpZero.b) {
        result.b = 1.0;
      }
      if (isExpZero.a) {
        result.a = 1.0;
      }
      // let isNaN = vec4<f32>(lessVec4F32(a, vec4<f32>(0.0))) * vec4<f32>(lessVec4F32(floor(b), b));
      let isNaN = vec4<f32>(lessVec4F32(a, vec4<f32>(0.0))) * vec4<f32>(lessVec4F32(floor(b), b));
      //{CHECK_NAN_SNIPPET_VEC4_WGSL}
      return result;
    }

    [[stage(compute), workgroup_size(16, 16, 1)]]
    fn main([[builtin(global_invocation_id)]] global_id : vec3<u32>) {
      let index : u32 = global_id.x;
      var result : f32 = firstMatrix.numbers[index] + secondMatrix.numbers[index];
      let sizeA : u32 = uniforms.size[0];
      // resultMatrix.numbers[index] = f32(global_id.x); //f32(uniforms.size[0]);
      //resultMatrix.numbers[index] = dottest4(vec4<f32>(1.0,1.0,1.0,1.0), vec4<f32>(1.0,1.0,1.0,2.0));

      // http://127.0.0.1:5500/index_wgsl.html
      // let a = vec4<f32>(1.0,1.0,1.0,1.0);
      // let b = vec4<f32>(1.0,1.0,1.0,1.0);
      // let ia = (round(a));
      // let ib = (round(b));
      // let cond = ib != vec4<i32>(0); //notEqual(ib, vec4<i32>(0));
      // let result = vec4<i32>(0);
      // let s = sign(a) * sign(b);
      // ptr_f32: ptr<function,f32>;
      
      // var ptr_f32 = 0.0;
      // var part = modf(100.1, &ptr_f32);
 
      // var ptr_vec4f32 = vec4<f32>(0.0);
      // var ptr2_vec4f32 = vec4<f32>(10.18);
      // var part_vec4f32 = modf(ptr2_vec4f32, &ptr_vec4f32);
      // resultMatrix.numbers[index] = ptr_vec4f32[0];

      // resultMatrix.numbers[index] = 11.0% 2.0;

      // let a = vec4<f32>(2.0, 3.0, 1.0, 1.0);
      // let b = a % vec4<f32> (2.0);
      // resultMatrix.numbers[index] = b[2];

      let a = vec4<f32>(2.0, 3.0, 1.0, 1.0);
      let b = vec4<f32>(3.0, 2.0, 1.0, 1.0);
      let c = binaryOperation(a, b);
      resultMatrix.numbers[index] = c[1];

    }
`;
