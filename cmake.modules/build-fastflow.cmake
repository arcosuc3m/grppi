if (NOT EXISTS ${CMAKE_BINARY_DIR}/fastflow)
  message("FastFlow not found. Installing...")

  ExternalProject_Add(fastflow
    SVN_REPOSITORY "svn://svn.code.sf.net/p/mc-fastflow/code"
    SVN_REVISION "514"
    PREFIX fastflow
    BUILD_IN_SOURCE
    UPDATE_COMMAND "svn up"
    CMAKE_CACHE_ARGS
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )  

endif()

set(FF_ROOT ${CMAKE_BINARY_DIR}/fastflow)
set(env(FF_ROOT) ${CMAKE_BINARY_DIR}/fastflow)
