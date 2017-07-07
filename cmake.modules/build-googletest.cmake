if (NOT EXISTS ${CMAKE_BINARY_DIR}/external/googletest)
  message("--googletest not found. Installing")

  ExternalProject_Add(googletest-release
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG "release-1.8.0"
    PREFIX external/googletest
    BUILD_IN_SOURCE
    UPDATE_COMMAND ""
    CMAKE_CACHE_ARGS
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DBUILD_GMOCK:BOOL=OFF
      -DBUILD_GTEST:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=Release
  )  

  ExternalProject_Add(googletest-debug
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG "release-1.8.0"
    PREFIX external/googletest
    BUILD_IN_SOURCE 
    UPDATE_COMMAND ""
    CMAKE_CACHE_ARGS
     -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
     -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
     -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
     -DBUILD_GMOCK:BOOL=OFF
     -DBUILD_GTEST:BOOL=ON
     -DCMAKE_BUILD_TYPE:STRING=Debug
     -DCMAKE_DEBUG_POSTFIX:STRING=d
 )
endif()

set(GTEST_ROOT ${CMAKE_BINARY_DIR}/external/googletest)
set(env(GTEST_ROOT) ${CMAKE_BINARY_DIR}/external/googletest)
