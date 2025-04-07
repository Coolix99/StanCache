cmake_minimum_required(VERSION 3.10)
project(StanCache)

set(CMAKE_CXX_STANDARD 17)

# Find Eigen (installed via apt)
find_package(Eigen3 REQUIRED)

# === Include Paths ===
include_directories(
    include                          # for stancache headers
    include/stancache
    include/nanoflann/include        # fix nanoflann substructure
    ${EIGEN3_INCLUDE_DIR}
)

# === Main Library ===
add_library(stancache
    src/gaussian_process.cpp
    src/kernel.cpp
    src/interpolator.cpp
)

target_include_directories(stancache PUBLIC include)

# === Tests ===
enable_testing()

add_executable(test_gp tests/test_gp.cpp)
target_link_libraries(test_gp stancache)
set_target_properties(test_gp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_test(NAME GaussianProcessTest COMMAND test_gp)

add_executable(test_interpolator tests/test_interpolator.cpp)
target_link_libraries(test_interpolator stancache)
set_target_properties(test_interpolator PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_test(NAME InterpolatorTest COMMAND test_interpolator)

# === Benchmark ===
add_executable(benchmark_gp benchmarks/benchmark_gp.cpp)
target_link_libraries(benchmark_gp stancache)
set_target_properties(benchmark_gp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
add_test(NAME BenchmarkGPTest COMMAND benchmark_gp)
