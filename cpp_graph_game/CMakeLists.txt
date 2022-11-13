cmake_minimum_required(VERSION 3.0)
set(CMAKE_BUILD_TYPE Debug)

set (CMAKE_CXX_STANDARD 17)

option(WITH_CUDA 								 "Enable CUDA support" 		ON)
option(USE_PROFILING             "Build with profiling"   OFF)
option(USE_RL                    "Build with reinforcement learning support"  ON)
option(BACKEND_TENSORRT          "Build with TensorRT support"  OFF)
option(BACKEND_TENSORRT_7        "Build with deprecated TensorRT 7 support"  OFF)
option(BACKEND_MXNET             "Build with MXNet backend (Blas/IntelMKL/CUDA/TensorRT) support"  OFF)
option(BACKEND_TORCH             "Build with Torch backend (CPU/GPU) support" ON)
option(BACKEND_OPENVINO          "Build with OpenVino backend (CPU/GPU) support" OFF)
option(BUILD_TESTS               "Build and run tests"  OFF)
option(USE_DYNAMIC_NN_ARCH       "Build with dynamic neural network architektur support"  ON)
# enable a single mode for different model input / outputs
option(MODE_CRAZYHOUSE           "Build with crazyhouse only support"  OFF)
option(MODE_CHESS                "Build with chess + chess960 only support"  OFF)
option(MODE_LICHESS              "Build with lichess variants support"  OFF)
option(MODE_OPEN_SPIEL           "Build with open_spiel environment support"  OFF)
option(MODE_BOARDGAMES           "Build with Fairy-Stockfish environment support for board games"   OFF)
option(MODE_XIANGQI              "Build with xiangqi only support" OFF)
option(MODE_STRATEGO             "Build Stratego with open_spiel environment support"  OFF)
option(MODE_HEX             		 "Build Hex graphs"  ON)
option(SEARCH_UCT                "Build with UCT instead of PUCT search"  OFF)
option(MCTS_STORE_STATES         "Build search by storing the state objects in each node. Results in higher memory usage but faster CPU runtime."  OFF)

add_definitions(-DIS_64BIT)


if (MODE_HEX)
		project(HexAra CXX)
		add_definitions(-DMODE_HEX)
endif ()

# -pg performance profiling flags
if (USE_PROFILING)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

if(DEFINED ENV{BLAZE_PATH})
    MESSAGE(STATUS "BLAZE_PATH set to: $ENV{BLAZE_PATH}")
    include_directories("$ENV{BLAZE_PATH}/include")
    include_directories("$ENV{BLAZE_PATH}")
else()
    MESSAGE(STATUS "BLAZE_PATH not set")
endif()

find_package( blaze REQUIRED HINTS ${BLAZE_PATH} )
add_library( blaze_target INTERFACE )
target_link_libraries( blaze_target INTERFACE blaze::blaze )

if (SEARCH_UCT)
    add_definitions(-DSEARCH_UCT)
endif()

if (MCTS_STORE_STATES)
    add_definitions(-DMCTS_STORE_STATES)
endif()

file(GLOB source_files
    "3rdparty/catch-2.91/*.hpp"
    "*.h"
    "*.cpp"
    "*.md"
		"CrazyAra/*.h"
		"CrazyAra/*.cpp"
    "CrazyAra/util/*.h"
    "CrazyAra/util/*.cpp"
    "CrazyAra/agents/*.cpp"
    "CrazyAra/agents/*.h"
    "CrazyAra/agents/config/*.cpp"
    "CrazyAra/agents/config/*.h"
    "CrazyAra/agents/util/*.cpp"
    "CrazyAra/agents/util/*.h"
    "CrazyAra/manager/*.cpp"
    "CrazyAra/manager/*.h"
    "CrazyAra/rl/*.cpp"
    "CrazyAra/rl/*.h"
    "hex_graph_game/*.h"
    "hex_graph_game/*.cpp"
)

file(GLOB_RECURSE CPP_PACKAGE_HEADERS
    "${CPP_PACKAGE_INCLUDE_DIR}/*.h"
    "${CPP_PACKAGE_INCLUDE_DIR}/*.hpp"
    )

include_directories("CrazyAra")

file(GLOB uci_files
	"CrazyAra/main/*.h"
	"CrazyAra/main/*.cpp"
    )

set(source_files
		${source_files}
		${uci_files}
		)


include_directories("3rdparty/catch-2.9.1")

# incude dirent library seperately because it's missing in the stdlib
if(MSVC OR MSYS OR MINGW)
    include_directories("3rdparty/dirent")
endif()

include_directories("CrazyAra/agents")
include_directories("CrazyAra/agents/config")
include_directories("hex_graph_game")



if (BACKEND_TORCH)
    message(STATUS "Enabled Torch Backend")
    find_package(Torch REQUIRED)
    add_definitions(-DTORCH)
endif()

if (USE_RL)
    message(STATUS "Enabled Reinforcement Learning functionality")
    if(DEFINED ENV{Z5_PATH})
        MESSAGE(STATUS "Z5_PATH set to: $ENV{Z5_PATH}")
    else()
        MESSAGE(STATUS "Z5_PATH not set")
    endif()
		if(DEFINED ENV{XTENSOR_PATH})
			MESSAGE(STATUS "XTENSOR_PATH set to: $ENV{XTENSOR_PATH}")
    else()
			MESSAGE(STATUS "XTENSOR_PATH not set")
    endif()
		if(DEFINED ENV{XTL_PATH})
			MESSAGE(STATUS "XTL_PATH set to: $ENV{XTL_PATH}")
    else()
			MESSAGE(STATUS "XTL_PATH not set")
    endif()
    include_directories($ENV{Z5_PATH})
    include_directories("3rdparty/json-3.7.0")
    include_directories($ENV{XTL_PATH})
    include_directories($ENV{XTENSOR_PATH})
    add_definitions(-DUSE_RL)
    add_definitions(-DDISABLE_UCI_INFO)
endif()


add_executable(${PROJECT_NAME} ${source_files})

if (USE_RL)
    # include filesystem (needed for z5)
    target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

find_package(Threads REQUIRED)

if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(${PROJECT_NAME} PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${PROJECT_NAME} "${CMAKE_THREAD_LIBS_INIT}")
endif()

# add target directory as library run path for unix systems
if(UNIX)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,./")
endif()

if (BACKEND_TORCH)
    target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
endif()

if (WITH_CUDA)
	find_package(CUDAToolkit REQUIRED)
endif()

list(APPEND CMAKE_PREFIX_PATH "/usr/local/include/libtorch")
find_package(Torch REQUIRED)
find_package(TorchScatter REQUIRED)
find_package(TorchSparse REQUIRED)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_range_for)
target_link_libraries(${PROJECT_NAME} TorchScatter::TorchScatter)
target_link_libraries(${PROJECT_NAME} TorchSparse::TorchSparse)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
if (WITH_CUDA)
target_link_libraries(${PROJECT_NAME} CUDA::cusparse)
endif()
