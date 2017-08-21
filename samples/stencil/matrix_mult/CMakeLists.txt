add_executable(matrix_mult main.cpp)

target_link_libraries(matrix_mult
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
