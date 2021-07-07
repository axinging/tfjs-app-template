export const computeShaderCodeWGSL = `
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

[[stage(compute), workgroup_size(16, 16, 1)]]

fn main([[builtin(local_invocation_id)]] localId : vec3<u32>,
      [[builtin(global_invocation_id)]] globalId : vec3<u32>) {

  let tileRow = localId.y * RowPerThread;
  let tileCol = localId.x;

  let globalRow = globalId.y * RowPerThread;
  let globalCol = globalId.x;
  let numTiles = (dimInner - 1u) / TileInner + 1u;

  var acc: array<vec4<f32>, 4>;
  var ACached : vec4<f32>;
  var BCached : array<vec4<f32>, 4>;

  // Without this initialization strange values show up in acc.
  for (var index = 0u; index < RowPerThread; index = index + 1u) {
      acc[index] = vec4<f32>(0.0);
  }

  // Loop over shared dimension.
  var globalColA = tileCol;
  let RowPerThreadB = TileInner / 16u;
  let tileRowB = localId.y * RowPerThreadB;
  for (var t = 0u; t < numTiles; t = t + 1u) {
      // Load one tile of A into local memory.
      for (var innerRow = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
          let inputRow = tileRow + innerRow;
          let inputCol = tileCol;
          
          mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, globalColA, globalId);//vec4<f32>(1.0);// 
	  
      }
      globalColA = globalColA + TileInner / ColPerThread;

      // Load one tile of B into local memory.
      for (var innerRow = 0u; innerRow < RowPerThreadB; innerRow = innerRow + 1u) {
          let inputRow = tileRowB + innerRow;
          let inputCol = tileCol;
          
          mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol, globalId);
      }

      workgroupBarrier();
     
      // Compute acc values for a single thread.
      for (var k = 0u; k < TileInner / ColPerThread; k = k + 1u) {
          BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
          BCached[1] = mm_Bsub[k * ColPerThread + 1u][tileCol];
          BCached[2] = mm_Bsub[k * ColPerThread + 2u][tileCol];
          BCached[3] = mm_Bsub[k * ColPerThread + 3u][tileCol];  

          for (var i = 0u; i < RowPerThread; i = i + 1u) {
              ACached = mm_Asub[tileRow + i][k];
              acc[i] = BCached[0] * ACached.x + acc[i];
              acc[i] = BCached[1] * ACached.y + acc[i];
              acc[i] = BCached[2] * ACached.z + acc[i];
              acc[i] = BCached[3] * ACached.w + acc[i];
          }
      }

      workgroupBarrier();
  }

  for (var innerRow = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
      mm_write(globalRow + innerRow,
               globalCol,
   	       acc[innerRow], globalId);
  }
}
    
`;
