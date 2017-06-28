# GOOGLE TEST----------------------------------------------------------------------------
include_directories( ${PROJECT_SOURCE_DIR} )

file(COPY ${CMAKE_SOURCE_DIR}/legacy_utest/txt DESTINATION "${CMAKE_BINARY_DIR}/legacy_utest/")

# Set policy to support newer version of CMake and GTest
# We will remove this when minimum version of CMake for GrPPI is move to 3.3 or
# higher
if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

find_package(GTest REQUIRED) 

include_directories(${GTEST_INCLUDE_DIRS})

if ( NOT (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") )
  # Add coverage options
  add_compile_options(--coverage)
endif()

set(NEW_TESTS_CPU 
    farm1 
    farm2                 farm3                         farm_example 
    filter_example        map1                          #map2 
    map3                  mapreduce1                    pipeline1 
    pipeline2             pipeline_example              pipeline+farm 
    pipeline+filter_test_ppi_example                     
    reduce1               reduce2                       stencil1 
    stream_reduce1        stream_reduce_example         stream_reduce_new
)


foreach( t ${NEW_TESTS_CPU} )
    add_executable(${t}_GT  ${CMAKE_SOURCE_DIR}/legacy_utest/${t}_GT.cpp)
    target_link_libraries(${t}_GT 
      ${GTEST_BOTH_LIBRARIES} 
      ${CMAKE_THREAD_LIBS_INIT} 
      ${TBB_LIBRARIES}
      ${Boost_LIBRARIES} 
      gcov)
    GTEST_ADD_TESTS(${t}_GT "" ${CMAKE_SOURCE_DIR}/legacy_utest/${t}_GT.cpp)
endforeach( t )
