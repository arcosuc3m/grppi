if (NOT EXISTS ${CMAKE_BINARY_DIR}/hayai)
  message("--hayai not found. Installing")

  ExternalProject_Add(hayai
    GIT_REPOSITORY "https://github.com/nickbruun/hayai.git"
    #GIT_TAG "master"
    PREFIX hayai
    BUILD_IN_SOURCE
    UPDATE_COMMAND ""
    CMAKE_CACHE_ARGS
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )
endif()

set(HAYAI_ROOT ${CMAKE_BINARY_DIR}/hayai)
