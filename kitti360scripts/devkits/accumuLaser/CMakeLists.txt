cmake_minimum_required(VERSION 3.0)

project(point_accumulator)

SET(CMAKE_BUILD_TYPE "RELEASE")

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# OPENCV
set(OPENCV_LIBS ${OPENCV_CORE_LIBRARY} ${OPENCV_CALIB3D_LIBRARY})

# include
include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(point_accumulator src/main.cpp src/commons.cpp src/accumulation.cpp src/utils.cpp)
target_link_libraries(point_accumulator ${OPENCV_LIBS} Eigen3::Eigen)
