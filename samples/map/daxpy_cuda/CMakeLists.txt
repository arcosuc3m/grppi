cmake_minimum_required(VERSION 3.5)

find_package(CUDA)

set (CMAKE_CXX_FLAGS "--std=c++14")
set (CUDA_PROPAGATE_HOST_FLAGS ON)

cuda_add_executable(daxpy main.cu)

target_link_libraries(daxpy
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
