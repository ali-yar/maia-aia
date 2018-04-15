cmake_minimum_required(VERSION 2.8)

project(retinal-vessel-segmentation)

find_package(OpenCV)

include_directories( ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

SET(PROJECT_SOURCES main.cpp utils.cpp vesseldetector.cpp)
SET(PROJECT_HEADERS utils.h vesseldetector.h)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
