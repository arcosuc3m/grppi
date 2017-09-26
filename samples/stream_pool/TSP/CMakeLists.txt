add_executable(tsp_problem main.cpp )

target_link_libraries(tsp_problem 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
