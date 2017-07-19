add_executable(sqrinv_seq_farm main.cpp )

target_link_libraries(sqrinv_seq_farm
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
