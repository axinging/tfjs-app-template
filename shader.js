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

void main() {
  int index = int(gl_GlobalInvocationID.x);
  resultMatrix.numbers[index] = firstMatrix.numbers[index] + secondMatrix.numbers[index];
}
`;