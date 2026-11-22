cmake_minimum_required(VERSION 3.14)
project(pricer_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(pricer INTERFACE)
target_include_directories(pricer INTERFACE include)
target_link_libraries(pricer INTERFACE Threads::Threads)

add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE pricer)

enable_testing()
add_executable(test_pricer tests/test_pricer.cpp)
target_link_libraries(test_pricer PRIVATE pricer)
add_test(NAME pricer_tests COMMAND test_pricer)
