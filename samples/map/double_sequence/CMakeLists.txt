add_executable( double_sequence main.cpp )

target_link_libraries( double_sequence 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )