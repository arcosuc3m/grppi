add_executable(print_power main.cpp )

target_link_libraries(print_power
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
