cmake_minimum_required(VERSION 3.9)
project(simplenet)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} .)

set(CMAKE_CXX_STANDARD 11)

add_executable(simplenet
        main.cpp)

target_link_libraries( simplenet ${OpenCV_LIBS} )