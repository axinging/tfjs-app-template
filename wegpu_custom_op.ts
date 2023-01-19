import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';

export function getMainHeaderString(): string;
export function getMainHeaderString(index: string): string;
export function getMainHeaderString(...params: string[]): string {
  let snippet: string;
  switch (params.length) {
    case 0:
      snippet = `fn main() `;
      break;
    case 1:
      snippet = `fn main(${params[0]} : i32)`;
      break;
    default:
      throw Error('Unreachable');
  }
  return snippet;
}

class KeypointProgram implements tfwebgpu.WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: {x: number[]};
    dispatch: [number, number, number];
    variableNames = ['A'];
    workgroupSize: [number, number, number];
    size = true;
  
    constructor(outputShape: number[]) {
      const workgroupSizeX = 128;
      this.workgroupSize = [workgroupSizeX, 1, 1];
      this.outputShape = outputShape;
      this.dispatchLayout = tfwebgpu.webgpu_util.flatDispatchLayout(this.outputShape);
      this.dispatch = tfwebgpu.webgpu_util.computeDispatch(
          this.dispatchLayout, this.outputShape, this.workgroupSize);
      this.shaderKey = 'keypointOp';
    }
  
    getUserCode(): string {
      return `
        ${getMainHeaderString('index')} {
          if (index < uniforms.size) {
            let a = getAByOutputIndex(index);
            setOutputAtIndex(index, a);
          }
        }
        `;
    }
  }

  export function keypointOp<T extends tf.Tensor>(x: T): T {
    const webgpuBackend = tf.backend() as tfwebgpu.WebGPUBackend;
    const program = new KeypointProgram(x.shape);
  
    const outInfo: tf.TensorInfo =
        webgpuBackend.runWebGPUProgram(program, [x], x.dtype);
    const value = tf.engine().makeTensorFromTensorInfo(outInfo) as T;
  
    return value;
  }

/*
javascript：
async function runCustom() {
  const input = tf.tensor([1, 2, 3, 4]);
  console.log(await keypointOp(input).data());
}
*/
