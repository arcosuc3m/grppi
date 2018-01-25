set(FF_FOUND FALSE)
if (NOT EXISTS ${CMAKE_BINARY_DIR}/fastflow)
  message("FastFlow not found. Installing...")

  ExternalProject_Add(fastflow
    GIT_REPOSITORY https://github.com/fastflow/fastflow.git
    #GIT_TAG "master"
    PREFIX fastflow
    BUILD_IN_SOURCE
    UPDATE_COMMAND ""
    CMAKE_CACHE_ARGS
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DBUILD_TESTS:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
  )
  set(FF_FOUND TRUE)
endif()

set(FFPATH ${CMAKE_BINARY_DIR}/fastflow)
