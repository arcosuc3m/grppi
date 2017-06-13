set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )

add_executable( arithmetic_add main.cpp )

target_link_libraries( arithmetic_add pthread tbb ${Boost_LIBRARIES} )