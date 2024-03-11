// import './cmake-build-release/retinaface-simd.js'
import wasm from './cmake-build-release/retinaface-simd.wasm?init'
import imgPath from './iyyi.png'

// const memory = new WebAssembly.Memory({ initial: 256, maximum: 256 * 1024 / 64 })
// let memoryPtr = 0
const instance = await wasm({
    wasi_snapshot_preview1: {
        proc_exit (code) {
            console.log('proc_exit', code)
        },
        fd_write (fd, iovs, iovs_len, nwritten) {
            // implement your own fd_write
            const view = new DataView(instance.exports.memory.buffer)
            const buffers = []
            for (let i = 0; i < iovs_len; i++) {
                const ptr = view.getUint32(iovs + i * 8, true)
                const len = view.getUint32(iovs + i * 8 + 4, true)
                buffers.push(new Uint8Array(instance.exports.memory.buffer, ptr, len))
            }
            const decoder = new TextDecoder()
            const str = buffers.map(b => decoder.decode(b)).join('')
            console.log(str)
            view.setUint32(nwritten, str.length, true)
            return 0
        },
        fd_close (fd) {
            console.log('fd_close', fd)
        },
        fd_seek (fd, offset, whence, newoffset) {
            console.log('fd_seek', fd, offset, whence, newoffset)
        },
    }
})

const obj = instance.exports
const memory = obj.memory


// -------

// const Module = window.Module = { }
// var yolov5wasm = 'cmake-build-release/retinaface-basic.wasm'
// var yolov5js = 'cmake-build-release/retinaface-basic.js'
//
// await fetch(yolov5wasm)
//     .then(response => response.arrayBuffer())
//     .then(buffer => {
//         Module.wasmBinary = buffer;
//         var script = document.createElement('script');
//         script.src = yolov5js;
//         document.body.appendChild(script);
//         return new Promise((resolve, reject) => {
//             script.onload = resolve;
//             script.onerror = reject;
//         })
//     });
//
// const obj = Module
// -------


const canvas = document.getElementById('canvas') as HTMLCanvasElement
canvas.width = 960
canvas.height = 960

const img = new Image()
img.src = imgPath
await new Promise((resolve, reject) => {
    img.onload = resolve
    img.onerror = reject
})

const ctx = canvas.getContext('2d')
const scale = Math.min(canvas.width / img.width, canvas.height / img.height)
ctx.drawImage(img, 0, 0, img.width * scale, img.height * scale)

const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
console.log(obj)
// const data = imageData.data
const dataPtr = obj._malloc(canvas.width * canvas.height * 3)

const data = new Uint8Array(canvas.width * canvas.height * 3)

for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
        const i = y * canvas.height + x
        data[i * 3] = imageData.data[i * 4]
        data[i * 3 + 1] = imageData.data[i * 4 + 1]
        data[i * 3 + 2] = imageData.data[i * 4 + 2]
    }
}


// const dataHeap = new Uint8ClampedArray(obj.HEAPU8.buffer, dataPtr, data.length)
// dataHeap.set(data)
// console.log(obj._detect(dataPtr, canvas.width, canvas.height))


// console.log(memory.buffer)
const dataHeap = new Uint8ClampedArray(memory.buffer, dataPtr, data.length)
dataHeap.set(data)
// console.log(dataHeap)
console.time('detect')
const ret = obj.detect(dataPtr, canvas.width, canvas.height)
console.timeEnd('detect')

const len = new Uint32Array(memory.buffer, ret, 1)[0]
const floats = 4 + 5 * 2 + 1
const retMem = new Float32Array(memory.buffer, ret + 4, len * floats)

const faces = []
for (let i = 0; i < len; i++) {
    const marklands = []
    for (let j = 0; j < 5; j++) {
        marklands.push([retMem[i * floats + 4 + j * 2], retMem[i * floats + 4 + j * 2 + 1]])
    }
    faces.push({
        rect: [retMem[i * floats], retMem[i * floats + 1], retMem[i * floats + 2], retMem[i * floats + 3]],
        marklands,
        socre: retMem[i * floats + 4 + 5 * 2]
    })
}
console.log(faces)

obj._free(dataPtr)
obj._free(ret)

faces.forEach(face => {
    ctx.strokeStyle = 'red'
    ctx.strokeRect(face.rect[0], face.rect[1], face.rect[2] - face.rect[0], face.rect[3] - face.rect[1])
    face.marklands.forEach(markland => {
        ctx.beginPath()
        ctx.arc(markland[0], markland[1], 2, 0, Math.PI * 2)
        ctx.fill()
    })
})

// window.ww = instance
// window.g=instance.exports
