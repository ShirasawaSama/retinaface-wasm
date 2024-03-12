const env = {
  wasi_snapshot_preview1: {
    proc_exit () { },
    fd_write () { return 0 },
    fd_close () { },
    fd_seek () { return -1 }
  }
}

// WXML:
// <button bindtap="onTap">Detect</button> <canvas id="canvas" type="2d" style="width: 100%; height: 100%;"></canvas>

Component({
  methods: {
    async onTap () {
      try {
        const { instance } = await WXWebAssembly.instantiate('/pages/index/retinaface-basic.wasm.br', env)

        const res = await wx.chooseMedia({ count: 1, mediaType: ['image'] })

        const imgUrl = res.tempFiles[0].tempFilePath

        const obj = instance.exports

        const query = wx.createSelectorQuery()
        const canvas = await new Promise<HTMLCanvasElement>((resolve) => {
          query.select('#canvas')
            .fields({ node: true, size: true })
            .exec((res: any) => {
              resolve(res[0].node)
            })
        })

        const ctx = canvas.getContext('2d')!

        canvas.width = 1024
        canvas.height = 1024

        const image = (canvas as any).createImage() as HTMLImageElement
        image.src = imgUrl
        await new Promise((resolve, reject) => {
          image.onload = resolve
          image.onerror = reject
        })

        const scale = Math.min(canvas.width / image.width, canvas.height / image.height)
        const width = (image.width * scale / 16 | 0) * 16
        const height = (image.height * scale / 16 | 0) * 16
        ctx.drawImage(image, 0, 0, width, height)
        const imageData = ctx.getImageData(0, 0, width, height)

        const dataPtr = obj._malloc(width * height * 3)
        const data = new Uint8ClampedArray(width * height * 3)

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const i = y * width + x
            data[i * 3] = imageData.data[i * 4]
            data[i * 3 + 1] = imageData.data[i * 4 + 1]
            data[i * 3 + 2] = imageData.data[i * 4 + 2]
          }
        }

        new Uint8ClampedArray(obj.memory.buffer, dataPtr, data.length).set(data)

        console.time('detect')
        const ret = obj.detect(dataPtr, width, height)
        console.timeEnd('detect')

        const len = new Uint32Array(obj.memory.buffer, ret, 1)[0]
        const floats = 4 + 5 * 2 + 1
        const retMem = new Float32Array(obj.memory.buffer, ret + 4, len * floats)

        const faces: Array<{ rect: number[], landmarks: number[][], socre: number }> = []
        for (let i = 0; i < len; i++) {
          const landmarks: number[][] = []
          for (let j = 0; j < 5; j++) {
            landmarks.push([retMem[i * floats + 4 + j * 2], retMem[i * floats + 4 + j * 2 + 1]])
          }
          faces.push({
            rect: [retMem[i * floats], retMem[i * floats + 1], retMem[i * floats + 2], retMem[i * floats + 3]],
            landmarks,
            socre: retMem[i * floats + 4 + 5 * 2]
          })
        }
        console.log(faces)

        obj._free(dataPtr)
        obj._free(ret)

        faces.forEach(face => {
          ctx.strokeStyle = 'red'
          ctx.strokeRect(face.rect[0], face.rect[1], face.rect[2] - face.rect[0], face.rect[3] - face.rect[1])
          face.landmarks.forEach(markland => {
            ctx.beginPath()
            ctx.arc(markland[0], markland[1], 2, 0, Math.PI * 2)
            ctx.fill()
          })
        })
      } catch (e) {
        console.error(e)
      }
    }
  }
})
