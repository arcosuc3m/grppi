# GOOGLE TEST----------------------------------------------------------------------------
include_directories( ${PROJECT_SOURCE_DIR} )
option( GTest_ENABLE "Activate Google Test" OFF)

file(COPY ${CMAKE_SOURCE_DIR}/unit_tests/txt DESTINATION "${CMAKE_BINARY_DIR}/unit_tests/")

# OMP
if(${OMP_ENABLE})
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()


if(GTest_ENABLE)

  find_package(GTest REQUIRED)
  include_directories(${GTEST_INCLUDE_DIRS})

  # Add coverage options
  add_compile_options(--coverage)

  set(NEW_TESTS_CPU 
      divideandconquer1     divideandconquer-fibonacci    farm1 
      farm2                 farm3                         farm_example 
      filter_example        map1                          map2 
      map3                  mapreduce1                    pipeline1 
      pipeline2             pipeline_example              pipeline+farm 
      pipeline+filter_test_ppi_example                     
      reduce1               reduce2                       stencil1 
      stream_reduce1        stream_reduce_example         stream_reduce_new
  )


  foreach( t ${NEW_TESTS_CPU} )
      add_executable(${t}_GT ${CMAKE_SOURCE_DIR}/unit_tests/${t}_GT.cpp)
      # The COMPILE_FLAGS are defined in the test refering to the enable_flag.hpp file
      target_link_libraries(${t}_GT ${GTEST_BOTH_LIBRARIES} gcov pthread tbb)
      target_link_libraries(${t}_GT ${Boost_LIBRARIES} )
      GTEST_ADD_TESTS(${t}_GT "" ${CMAKE_SOURCE_DIR}/unit_tests/${t}_GT.cpp)
  endforeach( t )

  add_executable(utest ${CMAKE_SOURCE_DIR}/unit_tests/polymorphic_execution.cpp)
  target_link_libraries(utest ${GTEST_BOTH_LIBRARIES} gcov pthread tbb)
  target_link_libraries(utest ${Boost_LIBRARIES} )
  GTEST_ADD_TESTS(utest "" ${CMAKE_SOURCE_DIR}/unit_tests/polymorphic_execution.cpp)
  


  set(NEW_TESTS_GPU
      bench_farm  farm4  map4  mapreduce2  stream_reduce2
  )

  # CUDA tests
  if(CUDA_FOUND)
    foreach( t ${NEW_TESTS_GPU} )
        CUDA_ADD_EXECUTABLE(${t}_GT ${CMAKE_SOURCE_DIR}/unit_tests/${t}_GT.cu)
        target_link_libraries(${t}_GT ${GTEST_BOTH_LIBRARIES} gcov pthread tbb)
        target_link_libraries(${t}_GT ${Boost_LIBRARIES} )
        GTEST_ADD_TESTS(${t}_GT "" ${CMAKE_SOURCE_DIR}/unit_tests/${t}_GT.cu)
    endforeach( t )
  endif(CUDA_FOUND)


  # To run coverage
  # 1. make
  # 2. make coverage_test_name_GT

  # # Coverage options
  find_program(LCOV_CMD lcov)
  find_program(GENHTML_CMD genhtml)

  foreach(t ${NEW_TESTS_CPU})
    ADD_CUSTOM_TARGET(coverage_${t}_GT
        ${LCOV_CMD} --directory . --zerocounters
        COMMAND ${t}_GT
        COMMAND ${LCOV_CMD} --directory . --capture --output-file mycov.info
        COMMAND ${LCOV_CMD} --remove mycov.info '/usr/*' 'utest/*' 
            --output-file mycov.info.cleaned
        COMMAND ${GENHTML_CMD} -o mycov mycov.info.cleaned --legend -s 
        COMMAND ${CMAKE_COMMAND} -E remove mycov.info mycov.info.cleaned
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Computing coverage and generating report"
    )
    # Open firefox with the result
    ADD_CUSTOM_COMMAND(TARGET coverage_${t}_GT POST_BUILD
        COMMAND firefox ./mycov/index.html &
    )
  endforeach( t )

endif(GTest_ENABLE)
