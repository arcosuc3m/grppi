add_executable(image_resolution main.cpp )

target_link_libraries(image_resolution
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
