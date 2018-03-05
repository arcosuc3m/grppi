add_executable(sqrinv_seq_farm_pipe main.cpp )

target_link_libraries(sqrinv_seq_farm_pipe
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
