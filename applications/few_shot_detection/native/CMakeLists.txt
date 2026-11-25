cmake_minimum_required(VERSION 3.24)
# Set before enabling CUDA so compiler detection uses the Thor architecture.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 110 CACHE STRING "CUDA architectures")
endif()
project(dinov3_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(NATIVE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../artifacts" CACHE PATH "Native artifact directory")

find_package(CUDAToolkit REQUIRED)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../cuda;$ORIGIN/../lib;${CUDAToolkit_LIBRARY_DIR}")

add_library(gpu_geometry SHARED gpu_geometry.cu)
target_link_libraries(gpu_geometry PRIVATE CUDA::cudart)
set_target_properties(gpu_geometry PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${NATIVE_OUTPUT_DIR}/cuda")
install(TARGETS gpu_geometry LIBRARY DESTINATION lib)
