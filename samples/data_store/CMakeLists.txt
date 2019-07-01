include_directories("${BOOST_INCLUDEDIR}")
include_directories("${PC_LIBZMQ_INCLUDE_DIRS}")

add_executable(data_test data_test.cpp )

target_link_libraries(data_test
  ${ZeroMQ_LIBRARY}
  ${Boost_LIBRARIES})

add_executable(port_test port_test.cpp )

target_link_libraries(port_test
  ${ZeroMQ_LIBRARY}
  ${Boost_LIBRARIES})
