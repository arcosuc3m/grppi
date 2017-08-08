add_executable(daxpy main.cpp )

target_link_libraries(daxpy
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
