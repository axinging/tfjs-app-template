export function getComputeShaderCodeGLSL(workGroupSize) {
  return `#version 450
layout (local_size_x = ${workGroupSize[0]},
  local_size_y = ${workGroupSize[1]},
  local_size_z = ${workGroupSize[2]}) in;

layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
    float numbers[];
} firstMatrix;

layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
    float numbers[];
} secondMatrix;

layout(std430, set = 0, binding = 2) buffer ResultMatrix {
    float numbers[];
} resultMatrix;

layout(std140, set = 0, binding = 3) uniform Uniforms {
  float NAN; ivec4 aShape; ivec4 bShape; ivec4 outShape;
};

float binaryOperation(float a, float b) {
  // if((a < 0.0) && (floor(b) < b)){
  //   return NAN;
  // }
  if (b == 0.0) {
    return 1.0;
  }

  return a+b;
}

void main() {
  resultMatrix.numbers[index] = binaryOperation(firstMatrix.numbers[index], secondMatrix.numbers[index]);
}
`;
}
