# Require git installed
find_package(Git)
if (NOT GIT_FOUND)
  message(ERROR "Git not found. Please install git")
endif()

# Require svn installed
find_package(Subversion)
if (NOT SUBVERSION_FOUND)
  message(ERROR "Subversion not found. Please install svn")
endif()

# Support use of add_external_project 
include(ExternalProject)

# Support custom generators
if(CMAKE_EXTRA_GENERATOR)
  set(BUILD_EXTERNAL_GENERATOR "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
else()
  set(BUILD_EXTERNAL_GENERATOR "${CMAKE_GENERATOR}" )
endif()

# Build packages if required version not found
include(${CMAKE_MODULE_PATH}/build-googletest.cmake)

# Build packages if required version not found
#include(${CMAKE_MODULE_PATH}/build-hayai.cmake)
