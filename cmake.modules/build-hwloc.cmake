if (NOT EXISTS ${CMAKE_BINARY_DIR}/external/hwloc)
  message("--hwloc not found. Installing")

  ExternalProject_Add(hwloc
    GIT_REPOSITORY "https://github.com/open-mpi/hwloc.git"
    GIT_TAG "v1.11"
    PREFIX external/hwloc
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ./autogen.sh && ./configure --prefix=<INSTALL_DIR>
#    CMAKE_COMMAND ""
#    CMAKE_CACHE_ARGS
#      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
#      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
#      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )  

endif()

set(HWLOC_ROOT ${CMAKE_BINARY_DIR}/external/hwloc)
set(env(HWLOC_ROOT) ${CMAKE_BINARY_DIR}/external/hwloc)
set(HWLOC_INCLUDE_DIR "${HWLOC_ROOT}/external/hwloc/include")
set(env(HWLOC_INCLUDE_DIR) "${HWLOC_ROOT}/external/hwloc/include")
set(HWLOC_LIBS "${HWLOC_ROOT}/external/hwloc/lib")
set(env(HWLOC_LIBS) "${HWLOC_ROOT}/external/hwloc/lib")
