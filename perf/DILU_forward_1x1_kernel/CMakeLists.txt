cmake_minimum_required(VERSION 3.18.4)
project(hotkernel_DILU_forward_1x1_kernel)

# Set MCC and MUSA path
set(MUSA_ROOT "/usr/local/musa" CACHE STRING "Set MUSA root directory")
set(MUSA_BIN "${MUSA_ROOT}/bin" CACHE STRING "Set MUSA bin directory")
# set(MUSA_BIN "/data/mingxu/mtcc/bin" CACHE STRING "Set MUSA bin directory")

# Set C and C++ compiler
set(MCC_C_COMPILER "${MUSA_BIN}/clang" CACHE STRING "Set MCC C Compiler path")
set(MCC_CXX_COMPILER "${MUSA_BIN}/clang++" CACHE STRING "Set MCC C++ Compiler path")
set(CMAKE_C_COMPILER ${MCC_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MCC_CXX_COMPILER})

# Set MUSA include and library path
set(MUSA_INC "${MUSA_ROOT}/include" CACHE STRING "Set MUSA_INC path")
set(MUSA_LIB "${MUSA_ROOT}/lib" CACHE PATH "Set MUSA_LIB path")
add_compile_options(-x musa --musa-path=${MUSA_ROOT})
add_link_options(-L${MUSA_LIB})

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



set(PROJECT_BUILD_DIR ${PROJECT_SOURCE_DIR}/build)

# set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
