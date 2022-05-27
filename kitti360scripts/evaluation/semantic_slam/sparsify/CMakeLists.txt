project(sparsify)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


# External dependency: PCL
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Settings.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not specified, using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions("-O2 -msse2 -msse3 -std=c++11")
endif()

# Evaluation executable.
add_executable(sparsify
    main.cpp
    utils.cpp
    utils.h
)
target_link_libraries(sparsify
    ${PCL_LIBRARIES}
)
