cmake_minimum_required(VERSION 3.24)
project(WatchDog)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
add_executable(WatchDog monitoring_threads.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(WatchDog PRIVATE ${OpenCV_LIBS} pthread)



