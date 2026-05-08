cmake_minimum_required(VERSION 3.20)
project(duotrigordle_solver CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options($<$<CONFIG:Release>:-O3>)

set(DATA_DIR "${CMAKE_SOURCE_DIR}/data" CACHE PATH "Path to wordlist data")
add_compile_definitions(DT_DATA_DIR="${DATA_DIR}")

add_library(dt_core STATIC core/src/feedback.cpp)
target_include_directories(dt_core PUBLIC core/include)
