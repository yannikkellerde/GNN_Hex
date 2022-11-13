set(CMAKE_BUILD_TYPE Debug)
list(APPEND CMAKE_PREFIX_PATH "/usr/local/include/libtorch")
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_torch_script)

option(WITH_CUDA "Enable CUDA support" ON)
# option(FOR_INFERENCE "" ON)
# add_definitions(-DFOR_INFERENCE)

include_directories("/usr/local/libtorch/include")
include_directories("/usr/local/libtorch")

if(DEFINED ENV{BLAZE_PATH})
    MESSAGE(STATUS "BLAZE_PATH set to: $ENV{BLAZE_PATH}")
    include_directories("$ENV{BLAZE_PATH}/include")
    include_directories("$ENV{BLAZE_PATH}")
else()
    MESSAGE(STATUS "BLAZE_PATH not set")
endif()

find_package( blaze REQUIRED HINTS ${BLAZE_PATH} )
if( blaze_FOUND )
     add_library( blaze_target INTERFACE )
     target_link_libraries( blaze_target INTERFACE blaze::blaze )
endif()

file(GLOB source_files
    "*.h"
    "*.cpp"
    "tests/*.h"
    "tests/*.cpp"
    "main/*.h"
    "main/*.cpp"
)

set(source_files
		${source_files}
		)

find_package(Torch REQUIRED)
find_package(TorchScatter REQUIRED)
find_package(TorchSparse REQUIRED)

if (WITH_CUDA)
find_package(CUDAToolkit REQUIRED)
endif()

add_executable(main ${source_files})
target_compile_features(main PUBLIC cxx_range_for)
target_link_libraries(main TorchScatter::TorchScatter)
target_link_libraries(main TorchSparse::TorchSparse)
target_link_libraries(main "${TORCH_LIBRARIES}")
if (WITH_CUDA)
target_link_libraries(main CUDA::cusparse)
endif()
set_property(TARGET main PROPERTY CXX_STANDARD 17)
