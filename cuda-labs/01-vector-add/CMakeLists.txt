cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build")
endif()

project(vector_add_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

foreach(target vector_add vector_add_bench)
  add_executable(${target} ${target}.cu)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME vector_add_example COMMAND vector_add)
  set_tests_properties(vector_add_example PROPERTIES PASS_REGULAR_EXPRESSION "c\\[11\\] = 33")
  add_test(NAME vector_add_direct COMMAND vector_add_bench 1003 256 direct)
  add_test(NAME vector_add_grid_stride COMMAND vector_add_bench 1048579 256 grid_stride)
endif()
