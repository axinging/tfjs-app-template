export function getComputeShaderCodeWGSL(workGroupSize) {
  return `
[[block]] struct Uniforms { NAN : u32; xShape : vec4<u32>; wShape : vec4<u32>; outShape : vec4<u32> ; 
     outShapeStrides: vec3<u32>; filterDims : vec2<u32>; pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>;
    dimAOuter : u32; dimBOuter : u32; dimInner : u32;};

  [[block]] struct Matrix0 {
      numbers: array<vec4<f32>>;
  };

  [[group(0), binding(0)]] var<storage, write> result : Matrix0;


  [[block]] struct Matrix1 {
    numbers: array<vec4<f32>>;
  };
  [[group(0), binding(1)]] var<storage, read> x : Matrix1;
  

  [[block]] struct Matrix2 {
    numbers: array<vec4<f32>>;
  };
  [[group(0), binding(2)]] var<storage, read> W : Matrix2;  
  [[group(0), binding(3)]] var<uniform> uniforms : Uniforms;
      
  fn setOutput(d0 : u32, d1 : u32, d2 : u32, d3 : u32, value : vec4<f32>) {
    result.numbers[0u] = value;
  }

      
  let dimAOuter = 9u;// outShape[1] * outShape[2];
  let dimBOuter = 64u;//outShape[3];
  let dimInner = 27u;//filterDims[0] * filterDims[1] * xShape[3];

  fn mm_readA(row : u32, col : u32, globalId : vec3<u32>) -> vec4<f32> {
    return vec4<f32>(1.0);
  }

  fn mm_readB(row : u32, col : u32, globalId : vec3<u32>) -> vec4<f32> {
    return vec4<f32>(1.0);
  }

  fn mm_write(row : u32, col : u32, valueInput : vec4<f32>, globalId : vec3<u32>) {
    var batch = globalId.z;
    var value = valueInput;
    if (row < dimAOuter && col * 4u < dimBOuter)
    {
      let outCoord = vec4<u32>(
        batch,
        row / 3u, //uniforms.outShape[2],
        row % 3u, // uniforms.outShape[2],
        col * 4u);
         
      setOutput(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
        value);
    }
  }
      
let RowPerThread = 4u;
let ColPerThread = 4u; // only support ColPerThread = 4
let TileAOuter = 64u;
let TileBOuter = 64u;
let TileInner = 64u;

var<workgroup> mm_Asub : array<array<vec4<f32>, 16>, 64>;
var<workgroup> mm_Bsub : array<array<vec4<f32>, 16>, 64>;

[[stage(compute), workgroup_size(${workGroupSize[0]}, ${workGroupSize[1]}, ${workGroupSize[2]})]]

fn main([[builtin(local_invocation_id)]] localId : vec3<u32>,
      [[builtin(global_invocation_id)]] globalId : vec3<u32>) {

}
    
`;
}