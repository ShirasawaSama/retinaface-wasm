import Retinaface, { getWasmFile, env, NcnnModel } from 'retinaface-wasm'
import imagePath from './R.jpg'

const wasm = await WebAssembly.instantiateStreaming(fetch('https://cdn.jsdelivr.net/npm/retinaface-wasm/wasm/' + getWasmFile()), env)
const retinaface = new Retinaface(wasm.instance)

const landmarksModel = new NcnnModel(
    wasm.instance,
    await fetch('https://cdn.jsdelivr.net/npm/retinaface-wasm/models/face_landmarker_param.txt').then(res => res.text()),
    await fetch('https://cdn.jsdelivr.net/npm/retinaface-wasm/models/face_landmarker_bin.json').then(res => res.arrayBuffer()),
    ['input_12'],
    [478 * 3]
)

const image = new Image()
image.src = imagePath
await new Promise((resolve, reject) => {
    image.onload = resolve
    image.onerror = reject
})

const [data, scale] = retinaface.processImage(image)
const result = retinaface.detect(data, scale)

const tmpCanvas = new OffscreenCanvas(256, 256)
const tmpCtx = tmpCanvas.getContext('2d')!

result.forEach((face) => {
    const width = face.rect[2] - face.rect[0]
    const height = face.rect[3] - face.rect[1]
    const scale = Math.min(256 / width, 256 / height)
    tmpCtx.clearRect(0, 0, 256, 256)
    tmpCtx.drawImage(image, face.rect[0], face.rect[1], width, height, 0, 0, width * scale, height * scale)
    const data = landmarksModel.inference(tmpCtx.getImageData(0, 0, 256, 256), 1 / 255)[0]
    const points = []
    for (let i = 0; i < 478; i++) {
        const x = face.rect[0] + data[i * 3] / scale
        const y = face.rect[1] + data[i * 3 + 1] / scale
        points.push([x, y, data[i * 3 + 2]])
    }
    console.log(points)
})

landmarksModel.close()
retinaface.close()
