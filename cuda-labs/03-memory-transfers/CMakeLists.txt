cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build")
endif()

project(memory_transfers_lab LANGUAGES CXX CUDA)

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
  add_test(NAME transfers_small COMMAND memory_transfers 1024)
  add_test(NAME transfers_irregular COMMAND memory_transfers 1003)
  set_tests_properties(transfers_small transfers_irregular PROPERTIES
    PASS_REGULAR_EXPRESSION "All transfer correctness checks passed")
  add_test(NAME transfers_invalid_size COMMAND memory_transfers 0)
  set_tests_properties(transfers_invalid_size PROPERTIES WILL_FAIL TRUE)
  add_test(NAME transfers_cold_pageable_first COMMAND memory_transfers 1003 pageable cold)
  add_test(NAME transfers_cold_pinned_first COMMAND memory_transfers 1024 pinned cold)
  set_tests_properties(transfers_cold_pageable_first transfers_cold_pinned_first
    PROPERTIES PASS_REGULAR_EXPRESSION "All transfer correctness checks passed")
endif()
