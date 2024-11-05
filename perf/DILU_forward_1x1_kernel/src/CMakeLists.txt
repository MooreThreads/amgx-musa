set(AmgxPath ${PROJECT_SOURCE_DIR}/../..)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${AmgxPath}/core/include
    ${AmgxPath}/base/include
    
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BUILD_DIR}/bin)

add_executable(dilu_forward_1x1_kernel DILU_forward_1x1_kernel.cpp main.cpp)
target_compile_options(dilu_forward_1x1_kernel PRIVATE 
  --offload-arch=mp_22
  "SHELL:-mllvm --mtgpu-tiny-offset-hint"
  "SHELL:-mllvm --mtgpu-load-store-opt"
)

target_link_libraries(dilu_forward_1x1_kernel 
    musart
    stdc++fs
)
