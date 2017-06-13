add_executable( arithmetic_add main.cpp )

target_link_libraries( arithmetic_add pthread tbb ${Boost_LIBRARIES} )