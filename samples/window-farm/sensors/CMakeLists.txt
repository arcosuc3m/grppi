add_executable(sensors main.cpp )

target_link_libraries(sensors
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
