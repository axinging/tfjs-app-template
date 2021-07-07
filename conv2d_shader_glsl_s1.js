export const computeShaderCodeGLSL = `


#version 450

int idiv(int a, int b, float sign) {
  int res = a / b;
  int mod = a % b;
  if (sign < 0. && mod != 0) {
    res -= 1;
  }
  return res;
}

// Checks whether coordinates lie within the bounds of the shape.
bool coordsInBounds(ivec4 coord, ivec4 shape) {
  return all(greaterThanEqual(coord, ivec4(0))) &&
      all(lessThan(coord, shape));
}

bool coordsInBounds(ivec3 coord, ivec3 shape) {
  return all(greaterThanEqual(coord, ivec3(0))) &&
      all(lessThan(coord, shape));
}

bool coordsInBounds(ivec2 coord, ivec2 shape) {
  return all(greaterThanEqual(coord, ivec2(0))) &&
      all(lessThan(coord, shape));
}


    layout (local_size_x = 16,
            local_size_y = 16,
            local_size_z = 1) in;
  

  layout(std430, set = 0, binding = 0) writeonly buffer ssbOut {
    vec4 result[];
  };


    layout(std430, set = 0, binding = 1) readonly buffer ssbx {
      vec4 x[];
    };
  

    layout(std430, set = 0, binding = 2) readonly buffer ssbW {
      vec4 W[];
    };
  

      layout(std140, set = 0, binding = 3) uniform Uniforms {
          float NAN; ivec4 xShape; ivec4 wShape; ivec4 outShape; ivec3 outShapeStrides; ivec2 filterDims, pad, stride, dilation;
      };
  

    bool isnan_custom(float val) {
      return (val > 0.0 || val < 0.0) ? false : val != 0.0;
    }

    bvec4 isnan_custom(vec4 val) {
      return bvec4(isnan_custom(val.x),
        isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
    }

    #define isnan(value) isnan_custom(value)
  

int getFlatIndex(int coord, int shape) {
  return coord;
}

int getFlatIndex(ivec2 coords, ivec2 shape) {
  return int(dot(coords, ivec2(shape.y, 1.)));
}

int getFlatIndex(ivec3 coords, ivec3 shape) {
  return int(dot(coords, ivec3(shape.y * shape.z, shape.z, 1.)));
}

int getFlatIndex(ivec4 coords, ivec4 shape) {
  return int(dot(coords, ivec4(
    shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1.)));
}


  ivec4 getCoordsFromFlatIndex(int index) {
    int d0 = index / outShapeStrides[0]; index -= d0 * outShapeStrides[0];int d1 = index / outShapeStrides[1]; index -= d1 * outShapeStrides[1];int d2 = index / outShapeStrides[2]; int d3 = index - d2 * outShapeStrides[2];
    return ivec4(d0,d1,d2,d3);
  }

ivec4 getOutputCoords() {
  int d3 =
      int(gl_GlobalInvocationID[0]);int index1 =
        int(gl_GlobalInvocationID[1]);int d1 = index1 / outShape[2];int d2 = index1 - d1 * outShape[2];int d0 =
      int(gl_GlobalInvocationID[2]);
return ivec4(d0,d1,d2,d3);}
void setOutput(int flatIndex, vec4 value) {
    result[flatIndex] = value;
  }
  void setOutput(int flatIndex, ivec4 value) {
    result[flatIndex] = vec4(value);
  }
      int getOutputFlatIndex(ivec4 coords) {
        return int(dot(coords, ivec4(
          outShapeStrides.x, outShapeStrides.y, outShapeStrides.z, 1)));
      }
      
    void setOutput(int d0, int d1, int d2, int d3, vec4 value) {
      int flatIndex = getOutputFlatIndex(ivec4(d0, d1, d2, d3));
      setOutput(flatIndex / 4, value);
    }
    void setOutput(int d0, int d1, int d2, int d3, ivec4 value) {
      int flatIndex = getOutputFlatIndex(ivec4(d0, d1, d2, d3));
      setOutput(flatIndex / 4, value);
    }
  

    vec4 getX(int d0, int d1, int d2, int d3) {
      return vec4(x[getFlatIndex(ivec4(d0,d1,d2,d3),
        xShape) / 4]);
    }
    
    vec4 getXAtOutCoords() {
      ivec4 coords = getOutputCoords();
      
      return x[getFlatIndex(ivec4(coords[0], coords[1], coords[2], coords[3]), xShape) / 4];
    }

    vec4 getXAtOutCoords(ivec4 coords) {
      
      return x[getFlatIndex(ivec4(coords[0], coords[1], coords[2], coords[3]), xShape) / 4];
    }
  

    vec4 getW(int d0, int d1, int d2, int d3) {
      return vec4(W[getFlatIndex(ivec4(d0,d1,d2,d3),
        wShape) / 4]);
    }
    
    vec4 getWAtOutCoords() {
      ivec4 coords = getOutputCoords();
      
      return W[getFlatIndex(ivec4(coords[0], coords[1], coords[2], coords[3]), wShape) / 4];
    }

    vec4 getWAtOutCoords(ivec4 coords) {
      
      return W[getFlatIndex(ivec4(coords[0], coords[1], coords[2], coords[3]), wShape) / 4];
    }
  

      
      
  vec4 mm_readA(int row, int col);
  vec4 mm_readB(int row, int col);
  void mm_write(int row, int col, vec4 value);

  int batch;
  int dimAOuter = 9;// outShape[1] * outShape[2];
  int dimBOuter = 64;//outShape[3];
  int dimInner = 27;//filterDims[0] * filterDims[1] * xShape[3];

  vec4 mm_readA(int row, int col) {
    int r = int(row), c = int(col * 4);
    if (r < dimAOuter && c < dimInner) {
    int outRow = r / outShape[2];
    int outCol = r % outShape[2];
    int WRow = c / (filterDims[1] * xShape[3]);
    int WCol = (c / xShape[3]) % filterDims[1];
    int inChCoord = c % xShape[3];
    ivec4 coord = ivec4(
        batch,
        outRow * stride[0] + dilation[0] * WRow - pad[0],
        outCol * stride[1] + dilation[1] * WCol - pad[1],
        inChCoord);
    vec4 resData = vec4(0.0);
    vec4 temp = vec4(0.0);
    int flatIndex = getFlatIndex(coord, xShape);
    int divBy4Remainder = flatIndex % 4;
    int divBy4Index = flatIndex / 4;
    vec4 curData = x[divBy4Index];
    if (divBy4Remainder == 0) {
        temp = curData;
    } else {
        // TODO: This could end up being a redundant load with another one in
        // the same shader invocation. Perhaps there's an opportunity for
        // optimization
        vec4 nextData = x[divBy4Index + 1];
        if (divBy4Remainder == 1) {
        temp = vec4(curData.yzw, nextData.x);
        } else if (divBy4Remainder == 2) {
        temp = vec4(curData.zw, nextData.xy);
        } else if (divBy4Remainder == 3) {
        temp = vec4(curData.w, nextData.xyz);
        }
    }
    
    resData = temp;
    if (WCol == (filterDims[1] - 1)) {
        coord = ivec4(
        coord.x, coord.y + 1, coord.z + 1 - filterDims[1], 0);
        int flatIndex = getFlatIndex(coord, xShape);
    int divBy4Remainder = flatIndex % 4;
    int divBy4Index = flatIndex / 4;
    vec4 curData = x[divBy4Index];
    if (divBy4Remainder == 0) {
        temp = curData;
    } else {
        // TODO: This could end up being a redundant load with another one in
        // the same shader invocation. Perhaps there's an opportunity for
        // optimization
        vec4 nextData = x[divBy4Index + 1];
        if (divBy4Remainder == 1) {
        temp = vec4(curData.yzw, nextData.x);
        } else if (divBy4Remainder == 2) {
        temp = vec4(curData.zw, nextData.xy);
        } else if (divBy4Remainder == 3) {
        temp = vec4(curData.w, nextData.xyz);
        }
    }
    
        if (inChCoord == 0) {
        resData = vec4(resData.xyz, temp.x);
        } else if (inChCoord == 1) {
        resData = vec4(resData.xy, temp.xy);
        } else {
        resData = vec4(resData.x, temp.xyz);
        }
    }
    
    return resData;
    } else {
        return vec4(0.0);
    };
  }

  vec4 mm_readB(int row, int col) {
    //return coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
    // W[row * dimBOuter / 4 + col] : vec4(0.0);
    return vec4(0.0);
  }

  void mm_write(int row, int col, vec4 value) {
    if (row < dimAOuter && col * 4 < dimBOuter)
    {
      ivec4 outCoord = ivec4(
        batch,
        row / 3, //outShape[2],
        row % 3, //outShape[2],
        col * 4);
      
      setOutput(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
        value);
    }
  }
  const int RowPerThread = 4;
  const int ColPerThread = 4; // only support ColPerThread = 4
  const int TileAOuter = 64; // int(gl_WorkGroupSize.y) * RowPerThread;
  const int TileBOuter = 64; // int(gl_WorkGroupSize.x) * ColPerThread;
  const int TileInner = 64; //TileBOuter;
  //shared vec4 mm_Asub[16][64];
  //shared vec4 mm_Bsub[16][64];
  // error X4586: The total amount of group shared memory (131072 bytes) exceeds the cs_5_1 limit of 32768 bytes
  shared vec4 mm_Asub[64][16];
  shared vec4 mm_Bsub[64][16];

  void main() {
    batch = int(gl_GlobalInvocationID.z);
    int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
    int tileCol = int(gl_LocalInvocationID.x);

    int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
    int globalCol = int(gl_GlobalInvocationID.x);

    int numTiles = (dimInner - 1) / TileInner + 1;

    vec4 acc[4];
    vec4 ACached;
    vec4 BCached[4];

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        acc[innerRow] = vec4(0.0);
    }

    // Loop over shared dimension.
    int globalColA = tileCol;
    const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
    int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;
    for (int t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          int inputRow = tileRow + innerRow;
          int inputCol = tileCol;

          mm_Asub[inputRow][inputCol] = vec4(1.0); //mm_readA(globalRow + innerRow, globalColA);
      }
      globalColA += TileInner / ColPerThread;

      // Load one tile of B into local memory.
      for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
          int inputRow = tileRowB + innerRow;
          int inputCol = tileCol;

          mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow,  globalCol);
      }

      barrier();

      // Compute acc values for a single thread.
      for (int k = 0; k < TileInner / ColPerThread; k++) {
        BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
        BCached[1] = mm_Bsub[k * ColPerThread + 1][tileCol];
        BCached[2] = mm_Bsub[k * ColPerThread + 2][tileCol];
        BCached[3] = mm_Bsub[k * ColPerThread + 3][tileCol];

        for (int i = 0; i < RowPerThread; i++) {
          ACached = mm_Asub[tileRow + i][k];
          acc[i] = BCached[0] * ACached.x + acc[i];
          acc[i] = BCached[1] * ACached.y + acc[i];
          acc[i] = BCached[2] * ACached.z + acc[i];
          acc[i] = BCached[3] * ACached.w + acc[i];
        }
      }
      barrier();
    }

    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
      mm_write(globalRow + innerRow,
        globalCol,
        acc[innerRow]);
    }
  }
    
`;
