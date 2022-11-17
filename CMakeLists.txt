cmake_minimum_required(VERSION 3.10.0)
set(PROJECT_NAME_STR grppi)
project(${PROJECT_NAME_STR})

# Path for additional CMake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules)

# Find packages
include(FetchContent)
# Handle new timestamp behaviour (see https://cmake.org/cmake/help/latest/policy/CMP0135.html)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

# Print processor and compiler
message(STATUS "Detected processor is " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})

if (NOT DEFINED CMAKE_CXX_STANDARD)
  message(STATUS "C++ standard not set")
  set(CMAKE_CXX_STANDARD 17)
endif ()
if (CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "Minimum standard is C++17")
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ standard set to ${CMAKE_CXX_STANDARD}")

add_compile_options(-Wall -Werror -pedantic -pedantic-errors -Wextra)

# Set specific options depending on compiler
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  if (NOT (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 3.9.0))
    message(FATAL_ERROR "Clang version " ${CMAKE_CXX_COMPILER_VERSION}
        " not supported. Upgrade to 3.9 or above.")
  endif ()
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  if (NOT (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 6.0))
    message(FATAL_ERROR "g++ version " ${CMAKE_CXX_COMPILER_VERSION}
        " not supported. Upgrade to 6.0 or above.")
  else ()
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 7.0)
      #g++ 7 warns in non C++17 for over-aligned new otherwise
      add_compile_options(-faligned-new)
    endif ()
  endif ()
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  message(WARNING "Intel compiler is not currently supported")
  # if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 16.0))
  #   message( FATAL_ERROR "Intel version " ${CMAKE_CXX_COMPILER_VERSION}
  #       " not supported. Upgrade to 16.0 or above.")
  # else()
  #   message( STATUS "C++ Compiler is Intel" )
  #   message( WARNING "WARNING: GrPPI has not been fully tested with Intel compiler." )
  # endif()
endif ()

message(STATUS "C++ Compiler is " ${CMAKE_CXX_COMPILER_ID}
    " version " ${CMAKE_CXX_COMPILER_VERSION})

# GrPPI library
include_directories(BEFORE SYSTEM "${CMAKE_SOURCE_DIR}/include")

# TBB Library
find_package(TBB)

if (TBB_FOUND)
  if (${CMAKE_CXX_STANDARD} STREQUAL 20)
    option(GRPPI_TBB_ENABLE "Require TBB library" OFF)
    set(GRPPI_TBB_ENABLE OFF CACHE BOOL "Require TBB library" FORCE)
    message(STATUS "TBB found, but not supported in C++20")
  else ()
    option(GRPPI_TBB_ENABLE "Require TBB library" ON)
    message(STATUS "TBB found")
  endif ()
else (TBB_FOUND)
  option(GRPPI_TBB_ENABLE "Require TBB library" OFF)
  set(GRPPI_TBB_ENABLE OFF CACHE BOOL "Require TBB library" FORCE)
  message(STATUS "TBB not found")
endif (TBB_FOUND)

message(STATUS "TBB: " ${GRPPI_TBB_ENABLE})
if (GRPPI_TBB_ENABLE)
  link_libraries(TBB::tbb)
  add_definitions(-DGRPPI_TBB)
endif (GRPPI_TBB_ENABLE)

# Threads Library
find_package(Threads REQUIRED)

# OMP library
find_package(OpenMP)
if (OPENMP_FOUND)
  option(GRPPI_OMP_ENABLE "Require OpenMP library" ON)
  message(STATUS "OpenMP found")
else (OPENMP_FOUND)
  option(GRPPI_OMP_ENABLE "Require OpenMP library" OFF)
  set(GRPPI_OMP_ENABLE OFF CACHE BOOL "Require OpenMP library" FORCE)
  message(STATUS "OpenMP not found")
endif (OPENMP_FOUND)

message(STATUS "OpenMP: " ${GRPPI_OMP_ENABLE})
if (GRPPI_OMP_ENABLE)
  add_definitions(-DGRPPI_OMP)
  add_compile_options(${OpenMP_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (GRPPI_OMP_ENABLE)

# Doxygen Documentation
option(GRPPI_DOXY_ENABLE "Require Doxygen library" OFF)
if (GRPPI_DOXY_ENABLE)
  add_subdirectory(doxy)
endif (GRPPI_DOXY_ENABLE)

# Installation
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/${CMAKE_PROJECT_NAME}"
    DESTINATION "include"
    FILES_MATCHING PATTERN "*.h")

# Example applications
option(GRPPI_EXAMPLE_APPLICATIONS_ENABLE "Sample tests" OFF)
if (GRPPI_EXAMPLE_APPLICATIONS_ENABLE)
  add_subdirectory(samples)
endif (GRPPI_EXAMPLE_APPLICATIONS_ENABLE)

# Unit Tests
enable_testing()
option(GRPPI_UNIT_TEST_ENABLE "Unit tests" ON)
if (GRPPI_UNIT_TEST_ENABLE)
  add_subdirectory(unit_tests)
endif (GRPPI_UNIT_TEST_ENABLE)
