add_executable( word_count main.cpp )

target_link_libraries( word_count 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
