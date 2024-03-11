cd emsdk
./emsdk install 3.1.55
./emsdk activate 3.1.55
source emsdk_env.sh

cd ..
wget https://github.com/Tencent/ncnn/releases/download/20240102/ncnn-20240102-webassembly.zip
unzip ncnn-20240102-webassembly.zip

mkdir build
cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
