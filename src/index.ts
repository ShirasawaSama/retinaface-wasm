export const isSimdSupported = (): boolean => {
  try {
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 30, 1, 28, 0, 65, 0,
      253, 15, 253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 186, 1, 26, 11
    ]))
  } catch {
    return false
  }
}

export const isBulkMemorySupported = (): boolean => {
  try {
    return WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 7, 1, 96,
      3, 127, 127, 127, 0, 3, 2, 1, 0, 5, 3, 1, 0, 1, 7, 14, 2, 3, 109, 101, 109, 2, 0, 4,
      102, 105, 108, 108, 0, 0, 10, 13, 1, 11, 0, 32, 0, 32, 1, 32, 2, 252, 11, 0, 11, 0, 10, 4,
      110, 97, 109, 101, 2, 3, 1, 0, 0
    ]))
  } catch {
    return false
  }
}

export const env = {
  wasi_snapshot_preview1: {
    proc_exit () { },
    fd_write () { return 0 },
    fd_close () { },
    fd_seek () { return -1 }
  }
}

export interface FaceObject {
  rect: [number, number, number, number]
  landmarks: [[number, number], [number, number], [number, number], [number, number], [number, number]]
  socre: number
}

export const createCanvas = (width: number, height: number) => {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height)
  } else {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    return canvas
  }
}

export const getWasmFile = (simd = isSimdSupported(), bulkMemory?: boolean): string => `retinaface-${
  simd
    ? 'simd'
    // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing
    : (bulkMemory == null ? isBulkMemorySupported() : bulkMemory) ? 'basic' : 'chrome57'
}.wasm`

export default class RetinaFace {
  public constructor (private readonly wasm: WebAssembly.Instance) { }

  public detect (imageData: ImageData, scale = 1, probThreshold = 0.75, nmsThreshold = 0.4): FaceObject[] {
    const dataPtr = (this.wasm.exports._malloc as (size: number) => number)(imageData.data.length)
    try {
      const memory = this.wasm.exports.memory as WebAssembly.Memory
      new Uint8ClampedArray(memory.buffer, dataPtr, imageData.data.length).set(imageData.data)

      const ret = (this.wasm.exports.detect as (ptr: number, width: number, height: number, p: number, t: number, pixelMode: number) => number)(
        dataPtr, imageData.width, imageData.height, probThreshold, nmsThreshold, 1
      )

      try {
        const len = new Uint32Array(memory.buffer, ret, 1)[0]
        if (len > 1000) throw new Error('Too many faces')
        const floats = 4 + 5 * 2 + 1
        const retMem = new Float32Array(memory.buffer, ret + 4, len * floats)

        const faces: FaceObject[] = []
        for (let i = 0; i < len; i++) {
          const landmarks: any = []
          for (let j = 0; j < 5; j++) {
            landmarks.push([retMem[i * floats + 4 + j * 2] / scale, retMem[i * floats + 4 + j * 2 + 1] / scale])
          }
          faces.push({
            rect: [retMem[i * floats] / scale, retMem[i * floats + 1] / scale, retMem[i * floats + 2] / scale, retMem[i * floats + 3] / scale],
            landmarks,
            socre: retMem[i * floats + 4 + 5 * 2]
          })
        }
        return faces
      } finally {
        (this.wasm.exports._free as (ptr: number) => void)(ret)
      }
    } finally {
      (this.wasm.exports._free as (ptr: number) => void)(dataPtr)
    }
  }

  public close (): void {
    (this.wasm.exports.destory as () => void)()
  }

  public processImage (image: HTMLImageElement | HTMLCanvasElement, rect?: { left?: number, top?: number, width?: number, height?: number }, width = 960, height = 960): [ImageData, number] {
    const r = { left: 0, top: 0, width: image.width, height: image.height, ...rect }
    const scale = Math.min(width / image.width, height / image.height)
    const canvas = createCanvas(width, height)
    const ctx = canvas.getContext('2d')! as CanvasRenderingContext2D
    ctx.drawImage(image, r.left, r.top, r.width, r.height, 0, 0, r.width * scale | 0, r.height * scale | 0)
    return [ctx.getImageData(0, 0, width, height), scale]
  }
}

export class NcnnModel {
  private readonly net: number
  private readonly extractNames: number[]
  private readonly extractMemory: Array<{ ptr: number, buffer: Float32Array }>

  public constructor (
    private readonly wasm: WebAssembly.Instance,
    params: string, bin: ArrayBuffer,
    extractNames: string[], extractMemory: number[]
  ) {
    if (!extractNames.length) throw new Error('extractNames should not be empty')
    if (extractMemory.length !== extractNames.length) throw new Error('extractMemory should have the same length as extractNames')

    const memory = this.wasm.exports.memory as WebAssembly.Memory
    const landmarkParamPtr = (wasm.exports._malloc as (size: number) => number)(params.length + 1)
    const landmarkParamHeap = new Uint8Array(memory.buffer, landmarkParamPtr, params.length + 1)
    landmarkParamHeap.set(Uint8Array.from(params.split('').map(x => x.charCodeAt(0))))
    landmarkParamHeap[params.length] = 0

    const ab = new Uint8Array(bin)
    const landmarkBinPtr = (wasm.exports._malloc as (size: number) => number)(ab.length)
    new Uint8Array(memory.buffer, landmarkBinPtr, ab.length).set(ab)

    this.net = (wasm.exports.create_net as (params: number, bin: number) => number)(landmarkParamPtr, landmarkBinPtr)
    this.extractNames = extractNames.map(name => {
      const extractNamePtr = (wasm.exports._malloc as (size: number) => number)(name.length + 1)
      const extractNameHeap = new Uint8Array(memory.buffer, extractNamePtr, name.length + 1)
      extractNameHeap.set(Uint8Array.from(name.split('').map(x => x.charCodeAt(0))))
      extractNameHeap[name.length] = 0
      return extractNamePtr
    })
    this.extractMemory = extractMemory.map(size => {
      const ptr = (wasm.exports._malloc as (size: number) => number)(size * 4)
      return { ptr, buffer: new Float32Array(memory.buffer, ptr, size) }
    })
  }

  public inference (imageData: ImageData, std = 1): Float32Array[] {
    const dataPtr = (this.wasm.exports._malloc as (size: number) => number)(imageData.data.length)
    try {
      const memory = this.wasm.exports.memory as WebAssembly.Memory
      new Uint8ClampedArray(memory.buffer, dataPtr, imageData.data.length).set(imageData.data)

      if (!(this.wasm.exports.inference as ((
        net: number, dataPtr: number, w: number, h: number,
        extractNamePtr: number, extractMemoryPtr: number, std: number, pixelMode: number) => number)
      )(this.net, dataPtr, imageData.width, imageData.height, this.extractNames[0], this.extractMemory[0].ptr, std, 1)) {
        throw new Error('inference failed')
      }

      return [this.extractMemory[0].buffer]
    } finally {
      (this.wasm.exports._free as (ptr: number) => void)(dataPtr)
    }
  }

  public close (): void {
    (this.wasm.exports.destory_net as (ptr: number) => void)(this.net)
    this.extractNames.forEach(ptr => { (this.wasm.exports._free as (ptr: number) => void)(ptr) })
    this.extractMemory.forEach(({ ptr }) => { (this.wasm.exports._free as (ptr: number) => void)(ptr) })
  }
}
