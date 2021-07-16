export function getComputeShaderCodeGLSL(workGroupSize) {
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
    float result[];
  };

  layout(std140, set = 0, binding = 3) uniform Uniforms {
    float NAN; ivec4 aShape; ivec4 bShape; ivec4 outShape;
  };
  
  void setOutput(int d0, int d1, int d2, int d3, float value) {
    result[0] = value;
  }
 
  void main() {
    uint M = aShape.y, N = bShape.y, K = aShape.z;

    // Thread identifiers
    uint globalRow = gl_GlobalInvocationID.x;
    uint globalCol = gl_GlobalInvocationID.y;

    // Compute a single element (loop over K)
    float acc = 0.0;
    for (uint k=0u; k<K; k++)
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    // Store the result
    result[globalCol*M + globalRow] = acc;
  }`;
}
