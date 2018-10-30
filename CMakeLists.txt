cmake_minimum_required(VERSION 3.9)
project(simplenet)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} .)

set(CMAKE_CXX_STANDARD 11)

add_library(simplenet SHARED
	./src/Layer.cpp ./src/Net.cpp ./src/Dataset.cpp ./src/funcs.cpp)

add_executable(train
	./src/train.cpp)

add_executable(evaluate
	./src/evaluate.cpp)

target_link_libraries( train simplenet pthread)#${OpenCV_LIBS} )
target_link_libraries( evaluate simplenet pthread ${OpenCV_LIBS} )

install( TARGETS simplenet train evaluate
           LIBRARY DESTINATION lib/
           RUNTIME DESTINATION bin/
         )
