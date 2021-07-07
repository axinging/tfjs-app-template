export const computeShaderCodeGLSL = `
#version 450
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
  
     
  void setOutput(int d0, int d1, int d2, int d3, vec4 value) {
    result[0] = value;
  }
 
  int batch;
  int dimAOuter = 9;// outShape[1] * outShape[2];
  int dimBOuter = 64;//outShape[3];
  int dimInner = 27;//filterDims[0] * filterDims[1] * xShape[3];

  vec4 mm_readA(int row, int col) {
    return vec4(1.0);
  }

  vec4 mm_readB(int row, int col) {
    return vec4(1.0);
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
    const int RowPerThreadB = TileInner / 16;
    int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;
    for (int t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          int inputRow = tileRow + innerRow;
          int inputCol = tileCol;

          mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, globalColA);
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
