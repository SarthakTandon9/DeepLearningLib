cmake_minimum_required(VERSION 3.14)
project(DeepFinanceDL VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable GPU support
option(USE_GPU "Enable GPU acceleration using CUDA" OFF)

# Find CUDA if GPU support is enabled
if(USE_GPU)
    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message(STATUS "CUDA found. GPU acceleration enabled.")
        set(HAS_GPU TRUE)
    else()
        message(WARNING "CUDA not found. GPU acceleration disabled.")
        set(HAS_GPU FALSE)
    endif()
else()
    set(HAS_GPU FALSE)
endif()

find_package(Eigen3 3.3 REQUIRED)
if (NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 library not found. Please make sure it is installed and available.")
endif()

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/DeepFinanceDL/core/tensor.cpp
    src/DeepFinanceDL/layers/dense.cpp
    src/DeepFinanceDL/activations/relu.cpp
    src/DeepFinanceDL/optimizers/sgd.cpp
    src/DeepFinanceDL/datasets/financial_dataset.cpp
    src/DeepFinanceDL/models/feedforward.cpp
    src/DeepFinanceDL/utils/logger.cpp
)


# Create library
add_library(DeepFinanceDL_lib ${SOURCES})

# Executable for examples
add_executable(simple_feedforward examples/simple_feedforward.cpp)
target_link_libraries(simple_feedforward DeepFinanceDL_lib)

# Enable testing
enable_testing()


