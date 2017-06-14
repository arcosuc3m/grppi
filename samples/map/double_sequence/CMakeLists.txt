add_executable( double_sequence main.cpp )

target_link_libraries( double_sequence pthread tbb ${Boost_LIBRARIES} )