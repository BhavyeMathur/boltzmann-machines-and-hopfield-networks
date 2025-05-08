cmake_minimum_required(VERSION 3.31)
project(main)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/vendor/eigen)
include_directories(${CMAKE_SOURCE_DIR}/vendor/cnpy)
include_directories(${CMAKE_SOURCE_DIR}/vendor/stb)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(main main.cpp
               vendor/cnpy/cnpy.cpp
               src/HopfieldNetwork.cpp
               src/HopfieldNetwork.h
               vendor/stb/stb_image_write.h
               vendor/stb/stb_image_write.cpp
               src/Util.cpp
               src/Util.h
               src/BoltzmannMachine.cpp
               src/BoltzmannMachine.h)
