add_executable(fibonacci main.cpp )

target_link_libraries(fibonacci
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
