cmake_minimum_required(VERSION 3.10)

project(DeepFinanceLib VERSION 1.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED)
if (NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 library not found. Please make sure it is installed and available.")
endif()

# Include directories
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sources and headers
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Add target
add_library(DeepFinanceLib STATIC ${SOURCES} ${HEADERS})

target_include_directories(DeepFinanceLib PUBLIC include ${EIGEN3_INCLUDE_DIR})

# Set target properties
target_compile_features(DeepFinanceLib PUBLIC cxx_std_17)

# Tests
enable_testing()

# Example usage message
message(STATUS "DeepFinanceLib version ${PROJECT_VERSION} configured successfully.")
