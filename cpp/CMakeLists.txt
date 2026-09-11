cmake_minimum_required(VERSION 3.10)
project(vision_fpv3 CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for GCC (CPU / ARM NEON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-O3 -Wall -Wextra)
endif()

# Find OpenCV & Python3
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui video tracking dnn)
find_package(Python3 COMPONENTS Development REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

set(SOURCES
    src/kalman_tracker.cpp
    src/camera_stream.cpp
    src/siamrpn_tracker.cpp
    src/vision_fpv3.cpp
    src/main.cpp
)

add_executable(vision_fpv3 ${SOURCES})
target_link_libraries(vision_fpv3 PRIVATE ${OpenCV_LIBS} Python3::Python)

set(BENCH_SOURCES
    src/kalman_tracker.cpp
    src/camera_stream.cpp
    src/siamrpn_tracker.cpp
    src/vision_fpv3.cpp
    src/benchmark_cpp_runner.cpp
)

add_executable(benchmark_cpp_runner ${BENCH_SOURCES})
target_link_libraries(benchmark_cpp_runner PRIVATE ${OpenCV_LIBS} Python3::Python)

message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
