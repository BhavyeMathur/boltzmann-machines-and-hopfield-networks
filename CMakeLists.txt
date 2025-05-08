cmake_minimum_required(VERSION 3.31)
project(hopfield)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/vendor/eigen)
include_directories(${CMAKE_SOURCE_DIR}/vendor/cnpy)
include_directories(${CMAKE_SOURCE_DIR}/vendor/stb)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(hopfield main.cpp
               vendor/cnpy/cnpy.cpp
               HopfieldNetwork.cpp
               HopfieldNetwork.h
               vendor/stb/stb_image_write.h
               vendor/stb/stb_image_write.cpp
               Util.cpp
               Util.h)
