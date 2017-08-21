add_executable(add_sequence main.cpp )

target_link_libraries(add_sequence 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
