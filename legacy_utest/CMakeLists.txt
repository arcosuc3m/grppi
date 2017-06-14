# GOOGLE TEST----------------------------------------------------------------------------
include_directories( ${PROJECT_SOURCE_DIR} )

file(COPY ${CMAKE_SOURCE_DIR}/legacy_utest/txt DESTINATION "${CMAKE_BINARY_DIR}/legacy_utest/")

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
    add_executable(${t}_GT EXCLUDE_FROM_ALL ${CMAKE_SOURCE_DIR}/legacy_utest/${t}_GT.cpp)
    target_link_libraries(${t}_GT 
      ${GTEST_BOTH_LIBRARIES} 
      ${CMAKE_THREAD_LIBS_INIT} 
      ${TBB_LIBRARIES}
      ${Boost_LIBRARIES} 
      gcov)
    GTEST_ADD_TESTS(${t}_GT "" ${CMAKE_SOURCE_DIR}/legacy_utest/${t}_GT.cpp)
endforeach( t )


set(NEW_TESTS_GPU
    bench_farm  farm4  map4  mapreduce2  stream_reduce2
)

# CUDA tests
if(CUDA_FOUND)
  foreach( t ${NEW_TESTS_GPU} )
      CUDA_ADD_EXECUTABLE(${t}_GT ${CMAKE_SOURCE_DIR}/legacy_utest/${t}_GT.cu)
      target_link_libraries(${t}_GT ${GTEST_BOTH_LIBRARIES} gcov pthread tbb)
      target_link_libraries(${t}_GT ${Boost_LIBRARIES} )
      GTEST_ADD_TESTS(${t}_GT "" ${CMAKE_SOURCE_DIR}/legacy_utest/${t}_GT.cu)
  endforeach( t )
endif(CUDA_FOUND)
