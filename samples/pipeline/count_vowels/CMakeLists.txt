add_executable(count_vowels main.cpp )

target_link_libraries(count_vowels 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
