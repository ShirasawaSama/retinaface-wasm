cd emsdk
./emsdk install 3.1.55
./emsdk activate 3.1.55
source emsdk_env.sh

cd ..
mkdir build
cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
