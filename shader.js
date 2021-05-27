export const computeShaderCode = `#version 450
//
bool isnan_custom(float val) {
  return (val > 0.0 || val < 0.0) ? false : val != 0.0;
}
bvec4 isnan_custom(vec4 val) {
  return bvec4(isnan_custom(val.x),
    isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
}
#define isnan(value) isnan_custom(value)
//

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
  if(isnan(NAN)) {
    resultMatrix.numbers[index] = NAN;
  } else {
    resultMatrix.numbers[index] = firstMatrix.numbers[index] + secondMatrix.numbers[index];
  }

}
`;