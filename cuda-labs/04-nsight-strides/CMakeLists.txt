cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build")
endif()

project(nsight_strides_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(strided_vector_add strided_vector_add.cu)
target_compile_options(strided_vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

include(CTest)
if(BUILD_TESTING)
  foreach(stride RANGE 1 5)
    add_test(NAME stride_${stride}
      COMMAND strided_vector_add 1280 ${stride})
    add_test(NAME bounds_stride_${stride}
      COMMAND strided_vector_add 253 ${stride})
  endforeach()
  add_test(NAME single_element COMMAND strided_vector_add 1 5)
  foreach(invalid IN ITEMS 0 -1 garbage 18446744073709551616)
    add_test(NAME invalid_n_${invalid} COMMAND strided_vector_add ${invalid} 1)
    set_tests_properties(invalid_n_${invalid} PROPERTIES WILL_FAIL TRUE)
  endforeach()
  foreach(invalid IN ITEMS 0 6 -1 garbage)
    add_test(NAME invalid_stride_${invalid} COMMAND strided_vector_add 1280 ${invalid})
    set_tests_properties(invalid_stride_${invalid} PROPERTIES WILL_FAIL TRUE)
  endforeach()
  add_test(NAME extra_argument COMMAND strided_vector_add 1280 1 extra)
  set_tests_properties(extra_argument PROPERTIES WILL_FAIL TRUE)
endif()
