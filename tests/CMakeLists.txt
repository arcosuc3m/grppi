
include_directories( ${PROJECT_SOURCE_DIR} )

# Set tests
set( TESTS_CPU_SEQ
     pipeline1              pipeline2              farm1
     farm2                  farm3                  stream_reduce1
     pipeline+farm          map1                   map2
     map3                   reduce1                reduce2
     stencil1               mapreduce1             divideandconquer1
     divideandconquer-fibonacci stream_reduce_new
     stream_iteration1
     stream_iteration_pipe
#     PPATtest               PPATfarm
# Examples in the Deliverable
     pipeline_example       farm_example           filter_example  
     stream_reduce_example  pipeline+filter_test_ppi_example
)

set( TESTS_CPU_OMP
     pipeline1              pipeline2              farm1
     farm2                  farm3                  stream_reduce1
     pipeline+farm          map1                   map2
     map3                   reduce1                reduce2
     stencil1               divideandconquer1      stream_reduce_new
 
     divideandconquer-fibonacci
# Examples in the Deliverable
     pipeline_example       farm_example           filter_example         
     stream_reduce_example  pipeline+filter_test_ppi_example 
)

set( TESTS_CPU_TBB                           
     pipeline1              pipeline2              farm1
     farm2                  farm3                  stream_reduce1 
     pipeline+farm          map1                   stream_reduce_new
     reduce1                reduce2
     stencil1 divideandconquer1
     divideandconquer-fibonacci
# Examples in the Deliverable
     pipeline_example       farm_example           filter_example
     stream_reduce_example  
)

set( TESTS_CPU_THR                           
     pipeline1              pipeline2              farm1
     farm2                  farm3                  stream_reduce1
     pipeline+farm          map1                   map2
     map3                   reduce1                reduce2
     stencil1               divideandconquer1      mapreduce1
     divideandconquer-fibonacci stream_reduce_new
     stream_iteration1
     stream_iteration_pipe
# Examples in the Deliverable
     pipeline_example       farm_example           filter_example
     stream_reduce_example  pipeline+filter_test_ppi_example
)

# Determine tests related to libraries. Sequential and Thread are always generated.
set ( EXE_LIB SEQ THR )

# TBB
if(${GRPPI_TBB_ENABLE})
    set(EXE_LIB ${EXE_LIB} TBB)
endif()

# OMP
if(${GRPPI_OMP_ENABLE})
    set(EXE_LIB ${EXE_LIB} OMP)
endif()


file(COPY ${CMAKE_SOURCE_DIR}/tests/txt DESTINATION "${CMAKE_BINARY_DIR}/tests/")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests/references ${CMAKE_BINARY_DIR}/tests/output)


# # Creation of the test executables including libraries if needed
foreach( m ${EXE_LIB} )
   foreach( t ${TESTS_CPU_${m}} )

      add_executable( ${t}_${m} ${t}.cpp )
      
      set_target_properties(${t}_${m} PROPERTIES COMPILE_FLAGS "-D${m}" )
      if ( ${m} STREQUAL THR )
         target_link_libraries( ${t}_${m} pthread )
      elseif ( ${m} STREQUAL TBB )
         target_link_libraries( ${t}_${m} tbb )
      endif()
      target_link_libraries( ${t}_${m} ${Boost_LIBRARIES} )
      
      # NOTE: pipeline4 is not here!!

      # Test creation
#       add_test(NAME "${t}_${m}" COMMAND "${t}_${m}" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tests/")
      if(${t} STREQUAL stream_iteration1)

      else()

      if(${m} STREQUAL SEQ)
          add_test( ${t}_${m}
                 ${CMAKE_COMMAND} -D TEST_PROGRAM=${CMAKE_BINARY_DIR}/tests/${t}_${m}${CMAKE_EXECUTABLE_SUFFIX}
                                  -D TEST_OUTPUT=references/reference_${t}_${m}.txt
                                  -P ${CMAKE_SOURCE_DIR}/runTestReference.cmake )
      else() 
            add_test( ${t}_${m}
                ${CMAKE_COMMAND} -D TEST_PROGRAM=${CMAKE_BINARY_DIR}/tests/${t}_${m}${CMAKE_EXECUTABLE_SUFFIX}
                           -D TEST_OUTPUT=output/output_${t}_${m}.txt -D TEST_REFERENCE=references/reference_${t}_SEQ.txt
                           -P ${CMAKE_SOURCE_DIR}/runTest.cmake )
      endif()
      endif()
      # Create reference files

      # Execute and compare tests with references in /test/tests_references
      #add_test( ${t}_${m}
      # ${CMAKE_COMMAND} -D TEST_PROGRAM=${CMAKE_BINARY_DIR}/tests/${t}_${m}${CMAKE_EXECUTABLE_SUFFIX}
      #            -D TEST_OUTPUT=output_${t}_${m}.txt -D TEST_REFERENCE=${CMAKE_SOURCE_DIR}/tests/tests_references/reference_${t}_${m}.txt
      #            -P ${CMAKE_SOURCE_DIR}/runTest.cmake )

      # Compare outputs with Sequential references

   endforeach( t )
endforeach( m )

# # CUDA tests
if(CUDA_FOUND)
    set( TESTS_GPU
        farm4
        map4
        # reduce3 # Error with the Reduce function parameter list
        bench_farm
        mapreduce2
        stream_reduce2
        #pipeline3 # It was not in the original CMakeFile - NOT working
    )
    foreach( t ${TESTS_GPU} )
        CUDA_ADD_EXECUTABLE( ${t} ${t}.cu )
        target_link_libraries( ${t} ${Boost_LIBRARIES} )
    endforeach( t )
endif(CUDA_FOUND)


#[[# GOOGLE TEST----------------------------------------------------------------------------

option( GTest_ENABLE "Activate Google Test" OFF)

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
      add_executable(${t}_GT ${CMAKE_SOURCE_DIR}/tests/${t}_GT.cpp)
      # The COMPILE_FLAGS are defined in the test refering to the enable_flag.hpp file
      target_link_libraries(${t}_GT ${GTEST_BOTH_LIBRARIES} gcov pthread tbb)
      target_link_libraries(${t}_GT ${Boost_LIBRARIES} )
      GTEST_ADD_TESTS(${t}_GT "" ${CMAKE_SOURCE_DIR}/tests/${t}_GT.cpp)
  endforeach( t )


  set(NEW_TESTS_GPU
      bench_farm  farm4  map4  mapreduce2  stream_reduce2
  )

  # CUDA tests
  if(CUDA_FOUND)
    foreach( t ${NEW_TESTS_GPU} )
        CUDA_ADD_EXECUTABLE(${t}_GT ${CMAKE_SOURCE_DIR}/tests/${t}_GT.cu)
        target_link_libraries(${t}_GT ${GTEST_BOTH_LIBRARIES} gcov pthread tbb)
        target_link_libraries(${t}_GT ${Boost_LIBRARIES} )
        GTEST_ADD_TESTS(${t}_GT "" ${CMAKE_SOURCE_DIR}/tests/${t}_GT.cu)
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
]]