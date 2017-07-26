add_executable(sort_sequence main.cpp )

target_link_libraries(sort_sequence 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
