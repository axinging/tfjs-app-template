// https://cnugteren.github.io/tutorial/pages/page4.html
export function getComputeShaderCodeGLSL1(workGroupSize) {
  return `
#version 450
  layout (local_size_x = ${workGroupSize[0]},
            local_size_y = ${workGroupSize[1]},
            local_size_z = ${workGroupSize[2]}) in;
  
  layout(std430, set = 0, binding = 0) readonly buffer ssbx {
    float A[];
  };

  layout(std430, set = 0, binding = 1) readonly buffer ssbW {
    float B[];
  };

  layout(std430, set = 0, binding = 2) writeonly buffer ssbOut {
    float C[];
  };

  layout(std140, set = 0, binding = 3) uniform Uniforms {
    float NAN; ivec4 aShape; ivec4 bShape; ivec4 outShape;
  };
  
  void setOutput(int d0, int d1, int d2, int d3, float value) {
    C[0] = value;
  }

  const uint TS = ${workGroupSize[0]};

  shared float Asub[TS][TS];
  shared float Bsub[TS][TS];
  void main() {
    uint M = aShape.y, N = bShape.y, K = aShape.z;

    // Thread identifiers
    const uint row =  gl_GlobalInvocationID.x; // Local row ID (max: TS)
    const uint col = gl_GlobalInvocationID.y; // Local col ID (max: TS)
    const uint globalRow = TS*gl_WorkGroupID.x + row; // Row ID of C (0..M)
    const uint globalCol = TS*gl_WorkGroupID.y + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B

    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const uint numTiles = K/TS;
    for (uint t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const uint tiledRow = TS*t + row;
        const uint tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];
 
        // Synchronise to make sure the tile is loaded
        barrier();
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        // Synchronise before loading the next tile
        barrier();
    }
 
    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
  }`;
}
