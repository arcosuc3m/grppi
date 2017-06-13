set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )

add_executable( double_sequence main.cpp )

target_link_libraries( double_sequence pthread tbb ${Boost_LIBRARIES} )