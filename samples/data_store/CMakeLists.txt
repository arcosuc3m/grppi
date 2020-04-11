include_directories("${BOOST_INCLUDEDIR}")
include_directories("${ZMQ_INCLUDE_DIR}")

add_executable(data_test data_test.cpp )

target_link_libraries(data_test
  ${ZMQ_LIBRARY}
  ${Boost_LIBRARIES})

add_executable(port_test port_test.cpp )

target_link_libraries(port_test
  ${ZMQ_LIBRARY}
  ${Boost_LIBRARIES})

add_executable(multi_queue_test multi_queue_test.cpp )

target_link_libraries(multi_queue_test)
