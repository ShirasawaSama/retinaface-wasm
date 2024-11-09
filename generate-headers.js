import fs from 'fs'

const mnet25ptBin = fs.readFileSync('models/retinaface_mnet25_bin.json')
const mnet25ptHex = '0x' + mnet25ptBin.toString('hex').match(/.{1,2}/g).join(', 0x').replace(/,$/, '')

const mnet25Params = fs.readFileSync('models/retinaface_mnet25_params.txt', 'utf8')

const mnet25OptC = `unsigned char FILES_mnet_25_opt_bin[] = {
  ${mnet25ptHex}
};
unsigned int FILES_LEN_mnet_25_opt_bin = ${mnet25ptBin.length};
const char* FILES_mnet_25_params = ${JSON.stringify(mnet25Params)};`

fs.writeFileSync('lib/model.h', mnet25OptC)
