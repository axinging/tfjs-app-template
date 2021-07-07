export const computeShaderCodeWGSL = `


fn idiv(a: i32, b: i32, sign: f32) -> i32 {
  var res: i32 = a / b;
  let mod: i32 = a % b;
  if (sign < 0. && mod != 0) {
    res = res - 1;
  }
  return res;
}
// Checks whether coordinates lie within the bounds of the shape.
fn coordsInBounds4D(coord: vec4<u32>, shape: vec4<u32>) -> bool {
  return all(coord >= vec4<u32>(0u, 0u, 0u, 0u)) &&
      all(coord < shape);
}
fn coordsInBounds3D(coord: vec3<u32>, shape: vec3<u32>) -> bool{
  return all(coord >= vec3<u32>(0u, 0u, 0u)) &&
      all(coord < shape);
}
fn coordsInBounds2D(coord: vec2<u32>, shape: vec2<u32>) -> bool {
  return all(coord >= vec2<u32>(0u, 0u)) &&
      all(coord < shape);
}

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
  

fn getFlatIndex1D(coord : u32, shape : u32) -> u32 {
  return coord;
}

fn getFlatIndex2D(coords : vec2<u32>, shape : vec2<u32>) -> u32 {
  return u32(dot(vec2<f32>(coords), vec2<f32>(f32(shape.y), 1.0)));
}

fn getFlatIndex3D(coords : vec3<u32>, shape : vec3<u32>) -> u32 {
  return u32(dot(vec3<f32>(coords), vec3<f32>(f32(shape.y) * f32(shape.z), f32(shape.z), 1.0)));
}

fn getFlatIndex4D(coords : vec4<u32>, shape : vec4<u32>) -> u32 {
  return u32(dot(vec4<f32>(coords), vec4<f32>(
      f32(shape.y) * f32(shape.z) * f32(shape.w), f32(shape.z) * f32(shape.w), f32(shape.w), 1.0)));
}


  fn getCoordsFromFlatIndex(index : u32) -> vec4<u32> {
    var index2 = index;let d0 = index2 / uniforms.outShapeStrides[0]; index2 = index2 - d0 * uniforms.outShapeStrides[0];let d1 = index2 / uniforms.outShapeStrides[1]; index2 = index2 - d1 * uniforms.outShapeStrides[1];let d2 = index2 / uniforms.outShapeStrides[2]; let d3 = index2 - d2 * uniforms.outShapeStrides[2];
    return vec4<u32>(d0,d1,d2,d3);
  }

fn getOutputCoords(globalId : vec3<u32>) -> vec4<u32> {
  let d3 = globalId[0];let index1 = globalId[1];let d1 = index1 / uniforms.outShape[2];let d2 = index1 - d1 * uniforms.outShape[2];let d0 = globalId[2];
return vec4<u32>(d0,d1,d2,d3); }
fn setOutputFlat(flatIndex : u32, value : vec4<f32>) {
    result.numbers[flatIndex] = value;
  }
  fn setOutputFlatI32(flatIndex : u32, value : vec4<i32>) {
    result.numbers[flatIndex] = vec4<f32>(value);
  }
      fn getOutputFlatIndex(coords : vec4<u32>) -> u32 {
        return u32(dot(vec4<f32>(coords), vec4<f32>(
          f32(uniforms.outShapeStrides.x), f32(uniforms.outShapeStrides.y), f32(uniforms.outShapeStrides.z), 1.0)));
      }
      
    fn setOutput(d0 : u32, d1 : u32, d2 : u32, d3 : u32, value : vec4<f32>) {
      let flatIndex = getOutputFlatIndex(vec4<u32>(d0, d1, d2, d3));
      setOutputFlat(flatIndex / 4u, value);
    }
    fn setOutputVectorI32(d0 : u32, d1 : u32, d2 : u32, d3 : u32, value : vec4<i32>) {
      let flatIndex = getOutputFlatIndex(vec4<u32>(d0, d1, d2, d3));
      setOutputFlatI32(flatIndex / 4u, value);
    }
  

    fn getX(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> vec4<f32> {
      return x.numbers[getFlatIndex4D(vec4<u32>(d0,d1,d2,d3),
        uniforms.xShape) / 4u];
    }
    
    fn getXAtOutCoordsByGlobalId(globalId : vec3<u32>) -> vec4<f32> {
      var coords = getOutputCoords(globalId);
      
      return x.numbers[getFlatIndex4D(vec4<u32>(coords[0u], coords[1u], coords[2u], coords[3u]), uniforms.xShape) / 4u];
    }

    fn getXAtOutCoordsByCoords(coordsIn : vec4<u32>) -> vec4<f32> {
      var coords = coordsIn;
      
      return x.numbers[getFlatIndex4D(vec4<u32>(coords[0u], coords[1u], coords[2u], coords[3u]), uniforms.xShape) / 4u];
    }
  

    fn getW(d0 : u32, d1 : u32, d2 : u32, d3 : u32) -> vec4<f32> {
      return W.numbers[getFlatIndex4D(vec4<u32>(d0,d1,d2,d3),
        uniforms.wShape) / 4u];
    }
    
    fn getWAtOutCoordsByGlobalId(globalId : vec3<u32>) -> vec4<f32> {
      var coords = getOutputCoords(globalId);
      
      return W.numbers[getFlatIndex4D(vec4<u32>(coords[0u], coords[1u], coords[2u], coords[3u]), uniforms.wShape) / 4u];
    }

    fn getWAtOutCoordsByCoords(coordsIn : vec4<u32>) -> vec4<f32> {
      var coords = coordsIn;
      
      return W.numbers[getFlatIndex4D(vec4<u32>(coords[0u], coords[1u], coords[2u], coords[3u]), uniforms.wShape) / 4u];
    }
  

      
    let dimAOuter = 9u;// outShape[1] * outShape[2];
    let dimBOuter = 64u;//outShape[3];
    let dimInner = 27u;//filterDims[0] * filterDims[1] * xShape[3];
    //var dimInner = 27u;//uniforms.dimInner;
  fn mm_readA(row : u32, col : u32, globalId : vec3<u32>) -> vec4<f32> {
    let r = row;
    let c = col * 4u;
    var batch = globalId.z;

    if (r < uniforms.dimAOuter && c < uniforms.dimInner) {
    let outRow = r / uniforms.outShape[2];
    let outCol = r % uniforms.outShape[2];
    let WRow = c / (uniforms.filterDims[1] * uniforms.xShape[3]);
    let WCol = (c / uniforms.xShape[3]) % uniforms.filterDims[1];
    let inChCoord = c % uniforms.xShape[3];
    var coord = vec4<u32>(
        batch,
        outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0],
        outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1],
        inChCoord);
    var resData = vec4<f32>(0.0);
    var temp = vec4<f32>(0.0);
    let flatIndex1 = getFlatIndex4D(coord, uniforms.xShape);
    let divBy4Remainder1 = flatIndex1 % 4u;
    let divBy4Index1 = flatIndex1 / 4u;
    let curData1 = x.numbers[divBy4Index1];
    if (divBy4Remainder1 == 0u) {
      temp = curData1;
    } else {
      // TODO: This could end up being a redundant load with another one in
      // the same shader invocation. Perhaps there's an opportunity for
      // optimization
      let nextData1 = x.numbers[divBy4Index1 + 1u];
      if (divBy4Remainder1 == 1u) {
        temp = vec4<f32>(curData1.yzw, nextData1.x);
      } elseif (divBy4Remainder1 == 2u) {
        temp = vec4<f32>(curData1.zw, nextData1.xy);
      } elseif (divBy4Remainder1 == 3u) {
        temp = vec4<f32>(curData1.w, nextData1.xyz);
      }
    }
  
    resData = temp;
    if (WCol == (uniforms.filterDims[1] - 1u)) {
      coord = vec4<u32>(
        coord.x, coord.y + 1u, coord.z + 1u - uniforms.filterDims[1], 0u);
        let flatIndex2 = getFlatIndex4D(coord, uniforms.xShape);
    let divBy4Remainder2 = flatIndex2 % 4u;
    let divBy4Index2 = flatIndex2 / 4u;
    let curData2 = x.numbers[divBy4Index2];
    if (divBy4Remainder2 == 0u) {
      temp = curData2;
    } else {
      // TODO: This could end up being a redundant load with another one in
      // the same shader invocation. Perhaps there's an opportunity for
      // optimization
      let nextData2 = x.numbers[divBy4Index2 + 1u];
      if (divBy4Remainder2 == 1u) {
        temp = vec4<f32>(curData2.yzw, nextData2.x);
      } elseif (divBy4Remainder2 == 2u) {
        temp = vec4<f32>(curData2.zw, nextData2.xy);
      } elseif (divBy4Remainder2 == 3u) {
        temp = vec4<f32>(curData2.w, nextData2.xyz);
      }
    }
  
      if (inChCoord == 0u) {
        resData = vec4<f32>(resData.xyz, temp.x);
      } elseif (inChCoord == 1u) {
        resData = vec4<f32>(resData.xy, temp.xy);
      } else {
        resData = vec4<f32>(resData.x, temp.xyz);
      }
    }
        
       return resData;
    }
    return vec4<f32>(0.0);
    
    }

fn mm_readB(row : u32, col : u32, globalId : vec3<u32>) -> vec4<f32> {
  //if(coordsInBounds2D(vec2<u32>(row, col * 4u), vec2<u32>(dimInner, dimBOuter))) {
   //return W.numbers[row * uniforms.dimBOuter / 4u + col];
  //}
  return vec4<f32>(0.0);

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
  // TODO: Remove it once the following bug is fixed.
  // https://bugs.chromium.org/p/tint/issues/detail?id=759
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
          
          mm_Asub[inputRow][inputCol] = vec4<f32>(1.0);// mm_readA(globalRow + innerRow, globalColA, globalId);//vec4<f32>(1.0);// 
	  
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
          BCached[0] =mm_Bsub[k * ColPerThread][tileCol];
          BCached[1] =mm_Bsub[k * ColPerThread + 1u][tileCol];
          BCached[2] =mm_Bsub[k * ColPerThread + 2u][tileCol];
          BCached[3] =mm_Bsub[k * ColPerThread + 3u][tileCol];  

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
               // acc[innerRow], globalId);
	       acc[innerRow], globalId);
  }
}
    
`;
