add_executable(discard_words main.cpp )

target_link_libraries(discard_words 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
