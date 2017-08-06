add_executable(add_sequences main.cpp )

target_link_libraries(add_sequences 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
