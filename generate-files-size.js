import fs from 'fs/promises'

const json = { }
const process = async dir => {
  const files = await fs.readdir(dir)
  for (const file of files) {
    const path = `${dir}/${file}`
    const stat = await fs.stat(path)
    if (stat.isDirectory()) {
      await process(path)
    } else {
      json[path] = stat.size
    }
  }
}
;(async () => {
  await Promise.all([
    process('wasm'),
    process('models')
  ])
  await fs.writeFile('files-size.json', JSON.stringify(json))
})()
