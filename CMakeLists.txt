cmake_minimum_required(VERSION 3.8)
project(segmentation)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(segmentation ${SOURCE_FILES})

find_package(OpenCV REQUIRED core imgproc highgui)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_objdetect)
target_link_libraries(segmentation ${OpenCV_LIBS})