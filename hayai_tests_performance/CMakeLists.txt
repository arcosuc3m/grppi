# HAYAI TEST----------------------------------------------------------------------------
option(GRPPI_HAYAI_TEST_ENABLE "Activate hayai performance tests" OFF)

if(GRPPI_HAYAI_TEST_ENABLE)
  
  # Set policy to support newer version of CMake and GTest
  # We will remove this when minimum version of CMake for GrPPI is move to 3.3 or
  # higher
  if(POLICY CMP0057)
    cmake_policy(SET CMP0057 NEW)
  endif()

  include_directories(${HAYAI_ROOT}/include/hayai)
  
  set(TESTS	map	reduce	mapreduce	stencil
  		divideandconquer	farm	pipeline
  )	
  
  # Determine system-dependant library paths.
  include(CheckLibraryExists)
  CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_RT_LIB)
  if (${NEED_RT_LIB})
    set(LIB_TIMING "rt")
  else (${NEED_RT_LIB})
    set(LIB_TIMING "")
  endif (${NEED_RT_LIB})

  # Unit testing should be made in debug mode
  set(CMAKE_BUILD_TYPE "Debug")

  set(PROJECT_TEST_NAME hayai_${PROJECT_NAME_STR})
  file(GLOB TEST_SRC_FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/*cpp)
      
  foreach( t ${TESTS} )
    add_executable( ${t} ${t}.cpp)
    target_link_libraries(${t}
    	${HAYAI_ROOT}/lib/libhayai_main.a
      	${LIB_TIMING} 
      	${TBB_LIBRARIES}
      	${CMAKE_THREAD_LIBS_INIT}
      	${Boost_LIBRARIES}
    )
  endforeach( t )

  add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
  target_link_libraries(${PROJECT_TEST_NAME} 
      ${HAYAI_ROOT}/lib/libhayai_main.a
      ${LIB_TIMING} 
      ${TBB_LIBRARIES}
      ${CMAKE_THREAD_LIBS_INIT}
      ${Boost_LIBRARIES}
  )
endif(GRPPI_HAYAI_TEST_ENABLE)
