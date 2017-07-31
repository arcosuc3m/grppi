
include_directories( ${PROJECT_SOURCE_DIR} )

# Set tests
set( TESTS_CPU_SEQ
     pipeline1              pipeline2              farm1
     farm2                  farm3                  
     pipeline+farm          
     stencil1               divideandconquer1
     divideandconquer-fibonacci 
     stream_iteration1      farm_sink
     stream_iteration_pipe
#     PPATtest               PPATfarm
# Examples in the Deliverable
     pipeline_example       farm_example           filter_example  
     pipeline+filter_test_ppi_example
)

set( TESTS_CPU_OMP
     pipeline1              pipeline2              farm1
     farm2                  farm3                  
     pipeline+farm          
     stencil1               divideandconquer1     
     farm_sink 		    
     divideandconquer-fibonacci
# Examples in the Deliverable
     pipeline_example       farm_example           filter_example         
     stream_reduce_example  pipeline+filter_test_ppi_example 
)

set( TESTS_CPU_TBB                           
     pipeline1              pipeline2              farm1
     farm2                  farm3                   
     pipeline+farm 
     stencil1 divideandconquer1
     divideandconquer-fibonacci
# Examples in the Deliverable
     pipeline_example       farm_example           filter_example
     stream_reduce_example       farm_sink 
)

set( TESTS_CPU_THR                           
     pipeline1              pipeline2              farm1
     farm2                  farm3                  
     pipeline+farm          
     stencil1               divideandconquer1    
     divideandconquer-fibonacci 
     stream_iteration1      farm_sink
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
         target_link_libraries( ${t}_${m} ${CMAKE_THREAD_LIBS_INIT} )
      elseif ( ${m} STREQUAL TBB )
         target_link_libraries( ${t}_${m} ${TBB_LIBRARIES} )
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
                                  -P ${CMAKE_SOURCE_DIR}/tests/runTestReference.cmake )
      else() 
            add_test( ${t}_${m}
                ${CMAKE_COMMAND} -D TEST_PROGRAM=${CMAKE_BINARY_DIR}/tests/${t}_${m}${CMAKE_EXECUTABLE_SUFFIX}
                           -D TEST_OUTPUT=output/output_${t}_${m}.txt -D TEST_REFERENCE=references/reference_${t}_SEQ.txt
                           -P ${CMAKE_SOURCE_DIR}/tests/runTest.cmake )
      endif()
      endif()
      # Create reference files

      # Execute and compare tests with references in /test/tests_references
      #add_test( ${t}_${m}
      # ${CMAKE_COMMAND} -D TEST_PROGRAM=${CMAKE_BINARY_DIR}/tests/${t}_${m}${CMAKE_EXECUTABLE_SUFFIX}
      #            -D TEST_OUTPUT=output_${t}_${m}.txt -D TEST_REFERENCE=${CMAKE_SOURCE_DIR}/tests/tests_references/reference_${t}_${m}.txt
      #            -P ${CMAKE_SOURCE_DIR}/tests/runTest.cmake )

      # Compare outputs with Sequential references

   endforeach( t )
endforeach( m )
