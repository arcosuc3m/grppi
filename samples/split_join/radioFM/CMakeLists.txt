add_executable(radio_equalizer main.cpp )

target_link_libraries(radio_equalizer 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
