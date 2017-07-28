add_executable(print_primes main.cpp )

target_link_libraries(print_primes 
  ${CMAKE_THREAD_LIBS_INIT} 
  ${TBB_LIBRARIES} 
  ${Boost_LIBRARIES} )
