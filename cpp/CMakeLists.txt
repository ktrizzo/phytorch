cmake_minimum_required(VERSION 3.16)
project(phytorch_wasm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PHYTORCH_BUILD_WASM    "Build the Emscripten/WASM module"     OFF)
option(PHYTORCH_BUILD_TESTS   "Build native sanity tests"            ON)

# ---------------------------------------------------------------------------
# Eigen3 (header-only). Prefer system install, fall back to FetchContent.
# ---------------------------------------------------------------------------
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "")
    set(BUILD_TESTING   OFF CACHE BOOL "")
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "")
    FetchContent_MakeAvailable(Eigen)
endif()

# ---------------------------------------------------------------------------
# Header-only fitting library.
# ---------------------------------------------------------------------------
add_library(phytorch INTERFACE)
target_include_directories(phytorch INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(phytorch INTERFACE Eigen3::Eigen)
target_compile_features(phytorch INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
# WASM module (built with `emcmake cmake ...`).
# ---------------------------------------------------------------------------
if(PHYTORCH_BUILD_WASM)
    if(NOT EMSCRIPTEN)
        message(FATAL_ERROR
            "PHYTORCH_BUILD_WASM=ON requires the Emscripten toolchain. "
            "Use: emcmake cmake -S . -B build-wasm -DPHYTORCH_BUILD_WASM=ON")
    endif()

    add_executable(phytorch_wasm bindings/wasm_bindings.cpp)
    target_link_libraries(phytorch_wasm PRIVATE phytorch)

    set_target_properties(phytorch_wasm PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "\
-O3 \
-lembind \
-sMODULARIZE=1 \
-sEXPORT_ES6=1 \
-sEXPORT_NAME=createPhytorchModule \
-sALLOW_MEMORY_GROWTH=1 \
-sENVIRONMENT=web \
-sFILESYSTEM=0 \
-sDISABLE_EXCEPTION_CATCHING=0")
endif()

# ---------------------------------------------------------------------------
# Native tests.
# ---------------------------------------------------------------------------
if(PHYTORCH_BUILD_TESTS AND NOT EMSCRIPTEN)
    enable_testing()

    add_executable(test_arrhenius tests/test_arrhenius.cpp)
    target_link_libraries(test_arrhenius PRIVATE phytorch)
    add_test(NAME arrhenius COMMAND test_arrhenius)

    add_executable(test_all tests/test_all.cpp)
    target_link_libraries(test_all PRIVATE phytorch)
    add_test(NAME all_models COMMAND test_all)

    add_executable(bench_native benchmarks/bench_native.cpp)
    target_link_libraries(bench_native PRIVATE phytorch)
    target_compile_options(bench_native PRIVATE -O3 -DNDEBUG)
endif()
