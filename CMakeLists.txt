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

add_library(dt_core STATIC
    core/src/feedback.cpp
    core/src/wordlists.cpp
    core/src/game_state.cpp
    core/src/strategy_greedy.cpp
    core/src/strategy_beam.cpp
    core/src/strategy_endgame.cpp
    core/src/simulator.cpp
)
target_include_directories(dt_core PUBLIC core/include)
target_link_libraries(dt_core PUBLIC OpenMP::OpenMP_CXX)

enable_testing()
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)
add_executable(dt_tests
    core/tests/test_feedback.cpp
    core/tests/test_wordlists.cpp
    core/tests/test_game_state.cpp
)
target_link_libraries(dt_tests PRIVATE dt_core GTest::gtest GTest::gtest_main)
add_test(NAME dt_tests COMMAND dt_tests)

add_executable(dt_bench core/bench/main.cpp)
target_link_libraries(dt_bench PRIVATE dt_core)

add_executable(dt_cli core/cli/main.cpp)
target_link_libraries(dt_cli PRIVATE dt_core)

find_package(nlohmann_json REQUIRED)
add_executable(dt_worker core/server/worker_main.cpp)
target_link_libraries(dt_worker PRIVATE dt_core nlohmann_json::nlohmann_json)

add_executable(dt_export_labels core/bench/export_turn_labels.cpp)
target_link_libraries(dt_export_labels PRIVATE dt_core)
