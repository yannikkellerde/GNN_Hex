list(APPEND CMAKE_PREFIX_PATH "/usr/local/include/libtorch")
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_torch_script)

find_package(Torch REQUIRED)
find_package(TorchScatter REQUIRED)
find_package(TorchSparse REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(test_torch_script test_torch_script.cpp)
target_compile_features(test_torch_script PUBLIC cxx_range_for)
target_link_libraries(test_torch_script TorchScatter::TorchScatter)
target_link_libraries(test_torch_script TorchSparse::TorchSparse)
target_link_libraries(test_torch_script "${TORCH_LIBRARIES}")
target_link_libraries(test_torch_script CUDA::cusparse)
set_property(TARGET test_torch_script PROPERTY CXX_STANDARD 17)

