add_executable(farm_GRPPI farm_GRPPI.cpp )

target_link_libraries(farm_GRPPI
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )

add_executable(map_GRPPI map_GRPPI.cpp )

target_link_libraries(map_GRPPI
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )

add_executable(reduce_GRPPI reduce_GRPPI.cpp )

target_link_libraries(reduce_GRPPI
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )

add_executable(stencil_GRPPI stencil_GRPPI.cpp )

target_link_libraries(stencil_GRPPI
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
  
if ( GRPPI_FF_ENABLE )

  add_executable(farm_FF farm_FF.cpp )

  target_link_libraries(farm_FF
    ${CMAKE_THREAD_LIBS_INIT} 
    ${TBB_LIBRARIES} 
    ${Boost_LIBRARIES} )
  
  add_executable(map_FF map_FF.cpp )

  target_link_libraries(map_FF
    ${CMAKE_THREAD_LIBS_INIT} 
    ${TBB_LIBRARIES} 
    ${Boost_LIBRARIES} )

  add_executable(reduce_FF reduce_FF.cpp )

  target_link_libraries(reduce_FF
    ${CMAKE_THREAD_LIBS_INIT} 
    ${TBB_LIBRARIES} 
    ${Boost_LIBRARIES} )
  
  add_executable(stencil_FF stencil_FF.cpp )

  target_link_libraries(stencil_FF
    ${CMAKE_THREAD_LIBS_INIT} 
    ${TBB_LIBRARIES} 
    ${Boost_LIBRARIES} )

endif ( GRPPI_FF_ENABLE )
