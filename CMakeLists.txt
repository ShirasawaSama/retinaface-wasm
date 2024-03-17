cmake_minimum_required(VERSION 3.13)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"
        CACHE STRING "Emsdk toolchain file")

project(retinaface_wasm)

set(CMAKE_CXX_STANDARD 20)

if (NOT WASM_FEATURE)
    set(WASM_FEATURE "basic")
endif ()

message(STATUS "WASM_FEATURE: ${WASM_FEATURE}")

if (WASM_FEATURE STREQUAL "chrome57")
    set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn-20240102-webassembly/basic/lib/cmake/ncnn")
else ()
    set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn-20240102-webassembly/${WASM_FEATURE}/lib/cmake/ncnn")
endif ()
find_package(ncnn REQUIRED)

set(CMAKE_EXECUTABLE_SUFFIX ".wasm")

add_executable(retinaface-${WASM_FEATURE} lib/main.cpp lib/model.h)

set(WASM_FLAGS "-Os -s INITIAL_MEMORY=256MB -s TOTAL_MEMORY=256MB -s FORCE_FILESYSTEM=1")

if (WASM_FEATURE STREQUAL "chrome57")
    set_target_properties(retinaface-${WASM_FEATURE} PROPERTIES COMPILE_FLAGS "${WASM_FLAGS} -s MIN_CHROME_VERSION=57")
    set_target_properties(retinaface-${WASM_FEATURE} PROPERTIES LINK_FLAGS "${WASM_FLAGS} -s MIN_CHROME_VERSION=57 -s WASM=1 -s STANDALONE_WASM --no-entry")
else ()
    set_target_properties(retinaface-${WASM_FEATURE} PROPERTIES COMPILE_FLAGS "${WASM_FLAGS}")
    set_target_properties(retinaface-${WASM_FEATURE} PROPERTIES LINK_FLAGS    "${WASM_FLAGS} -s WASM=1 -s STANDALONE_WASM --no-entry")
endif ()

target_link_libraries(retinaface-${WASM_FEATURE} ncnn)
