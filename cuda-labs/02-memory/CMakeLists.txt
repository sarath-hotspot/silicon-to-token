cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build")
endif()

project(memory_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB lab_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach(source IN LISTS lab_sources)
  get_filename_component(target "${source}" NAME_WE)
  add_executable("${target}" "${source}")
  target_compile_options("${target}" PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME memory_round_trip COMMAND memory)
  set_tests_properties(memory_round_trip PROPERTIES
    PASS_REGULAR_EXPRESSION "Memory round-trip check passed")
endif()
