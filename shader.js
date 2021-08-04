export const computeShaderCode = `#version 450
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
  float NAN; int sizeA; int sizeB; 
};

float binaryOperation(float a, float b) {
  // if((a < 0.0) && (floor(b) < b)){
  //   return NAN;
  // }
  if (b == 0.0) {
    return 1.0;
  }

  return (round(mod(b, 2.0)) != 1) ?
     pow(abs(a), b) : sign(a) * pow(abs(a), b);
}

void main() {
  // (-3.0, -3.0)
  // with: 0.03703703731298447;
  // without: -0.03703703731298447;

  // (-2.0, -3.0)
  // with: 0.125
  // without: -0.125

  // (-4.0, -3.0)
  // with: 0.015625
  // without: -0.015625
  int index = int(gl_GlobalInvocationID.x);
  int a = -10;
   resultMatrix.numbers[index] = 1.0/1e-20;// binaryOperation(-4.0, -3.0);
  // resultMatrix.numbers[index] = 1.0/ pow(10.0, -20.0);
}
`;
