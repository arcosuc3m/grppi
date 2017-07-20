add_executable(chunk_sum main.cpp )

target_link_libraries(chunk_sum
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
