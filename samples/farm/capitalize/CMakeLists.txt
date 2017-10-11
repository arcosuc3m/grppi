add_executable(capitalize main.cpp )

target_link_libraries(capitalize
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
