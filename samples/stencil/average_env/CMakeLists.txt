add_executable(average_env main.cpp)

target_link_libraries(average_env
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
