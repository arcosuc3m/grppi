if (NOT EXISTS ${CMAKE_BINARY_DIR}/googletest-download)
  message("--googletest not found. Installing")
  
  ExternalProject_Add(googletest
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG "master"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-src"
    BINARY_DIR "${CMAKE_BINARY_DIR}/googletest-build"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
  )
endif()

set(GTEST_ROOT ${CMAKE_BINARY_DIR}/external/googletest)
set(env(GTEST_ROOT) ${CMAKE_BINARY_DIR}/external/googletest)
