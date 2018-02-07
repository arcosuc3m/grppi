set(FF_FOUND FALSE)
if(NOT EXISTS ${CMAKE_BINARY_DIR}/fastflow)
  message("--FastFlow not found. Installing...")

  ExternalProject_Add(fastflow
   GIT_REPOSITORY https://github.com/fastflow/fastflow.git
    GIT_TAG "479817cf2dac93f732427693fb654f9f51b316f5"
    PREFIX fastflow
    BUILD_IN_SOURCE 0
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
  )
  set(FF_FOUND TRUE)
endif()

set(FFPATH ${CMAKE_BINARY_DIR}/fastflow/src)
