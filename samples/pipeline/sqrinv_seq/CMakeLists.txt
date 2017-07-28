add_executable(sqrinv_seq main.cpp )

target_link_libraries(sqrinv_seq 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
