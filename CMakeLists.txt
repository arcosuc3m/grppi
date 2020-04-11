cmake_minimum_required (VERSION 3.2.0)
set(PROJECT_NAME_STR grppi)
project(${PROJECT_NAME_STR})

# Path for additional CMake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules)

# Print processor and compiler 
message(STATUS "Detected processor is " ${CMAKE_SYSTEM_PROCESSOR} )
message(STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID} )

set(CMAKE_CXX_STANDARD 14)

add_compile_options(-Wall -Werror -pedantic -pedantic-errors -Wextra)

# Set specific options depending on compiler
if ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" )
  if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 3.9.0))
    message(FATAL_ERROR "Clang version " ${CMAKE_CXX_COMPILER_VERSION} 
       " not supported. Upgrade to 3.9 or above.")
  else()
    message( STATUS "C++ Compiler is Clang" )
  endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 6.0))
    message(FATAL_ERROR "g++ version " ${CMAKE_CXX_COMPILER_VERSION} 
        " not supported. Upgrade to 6.0 or above.")
  else()
    message( STATUS "C++ Compiler is GNU")
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 7.0)
      #g++ 7 warns in non C++17 for over-aligned new otherwise
      add_compile_options(-faligned-new)
    endif()
  endif()
elseif ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
  message(FATAL_ERROR "Intel compiler is not currently supported")   
# if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 16.0))
#   message( FATAL_ERROR "Intel version " ${CMAKE_CXX_COMPILER_VERSION} 
#       " not supported. Upgrade to 16.0 or above.")
# else()
#   message( STATUS "C++ Compiler is Intel" )
#   message( WARNING "WARNING: GrPPI has not been fully tested with Intel compiler." )
# endif()
else()
  message( FATAL_ERROR "Unsupported compiler: " 
      ${CMAKE_CXX_COMPILER_ID} 
      ${CMAKE_CXX_COMPILER_VERSION})
endif()

# GrPPI library
include_directories("${CMAKE_SOURCE_DIR}/include")

# TBB Library
find_package( TBB ) 
if( TBB_FOUND )
  option(GRPPI_TBB_ENABLE "Require TBB library" ON)
  message(STATUS "TBB found")
else( TBB_FOUND )
  option(GRPPI_TBB_ENABLE "Require TBB library" OFF )
  set(GRPPI_TBB_ENABLE OFF CACHE BOOL "Require TBB library" FORCE)
  message(STATUS "TBB not found")
endif( TBB_FOUND )

message( STATUS "TBB: " ${GRPPI_TBB_ENABLE} )
if( GRPPI_TBB_ENABLE )
  include_directories(${TBB_INCLUDE_DIRS} )
  add_definitions(-DGRPPI_TBB)
endif( GRPPI_TBB_ENABLE )

# Threads Library
find_package(Threads REQUIRED)

# OMP library
find_package(OpenMP)
if (OPENMP_FOUND) 
  option( GRPPI_OMP_ENABLE "Require OpenMP library" ON )
  message(STATUS "OpenMP found")
else(OPENMP_FOUND)
  option( GRPPI_OMP_ENABLE "Require OpenMP library" OFF )
  set(GRPPI_OMP_ENABLE OFF CACHE BOOL "Require OpenMP library" FORCE)
  message(STATUS "OpenMP not found")
endif (OPENMP_FOUND)

message( STATUS "OpenMP: " ${GRPPI_OMP_ENABLE})
if( GRPPI_OMP_ENABLE )
  add_definitions(-DGRPPI_OMP)
  add_compile_options(${OpenMP_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif( GRPPI_OMP_ENABLE )

#FastFlow library
find_package(FF)
if (FF_FOUND)
  option( GRPPI_FF_ENABLE "Require FastFlow library" ON)
  message(STATUS "FastFlow found")
else()
  option(GRPPI_FF_ENABLE "Require FastFlow library" OFF)
  set(GRPPI_FF_ENABLE OFF CACHE BOOL "Require FastFlow library" FORCE)
  message(STATUS "FastFlow not found")
  message(STATUS "  GRPPI_FF_ENABLE has been set to OFF")
  message(STATUS "  To enable it, please, first install FastFlow")
  message(STATUS "  https://github.com/fastflow/fastflow")
endif()

message( STATUS "FastFlow: " ${GRPPI_FF_ENABLE})
if(GRPPI_FF_ENABLE)
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(GRPPI_FF_ENABLE OFF CACHE BOOL "Require Fastflow library" FORCE)
    message(WARNING "Fastflow is not currently supported in Clang")
  else()
    add_definitions(-DGRPPI_FF)
    include_directories(${FF_INCLUDE_DIRS})
  endif()
endif(GRPPI_FF_ENABLE)

# BOOST library
find_package(Boost 1.66.0 COMPONENTS serialization REQUIRED)
if ( Boost_FOUND )
  option( GRPPI_BOOST_ENABLE "Require BOOST library" ON )
  link_directories(${Boost_LIBRARY_DIRS})
else( Boost_FOUND )
  option( GRPPI_BOOST_ENABLE "Require BOOST library" OFF )
endif( Boost_FOUND) 

find_package(ZeroMQ REQUIRED)

# libzmq autotools install: fallback to pkg-config
#if(NOT ZMQ_FOUND)
#    message(STATUS "CMake libzmq package not found, trying again with pkg-config (normal install of zeromq)")
#   find_package(ZeroMQ_pkg-config REQUIRED)
#endif()

# TODO "REQUIRED" above should already cause a fatal failure if not found, but this doesn't seem to work
if(NOT ZMQ_FOUND)
    message(FATAL_ERROR "ZeroMQ was not found, neither as a CMake package nor via pkg-config")
endif()

#if (ZMQ_FOUND AND (NOT TARGET libzmq OR NOT TARGET libzmq-static))
#  message(FATAL_ERROR "ZeroMQ version not supported!")
#endif()

message( ${ZMQ_INCLUDE_DIR} )
include_directories(${ZMQ_INCLUDE_DIR} )

# Doxygen Documentation
option( GRPPI_DOXY_ENABLE "Require Doxygen library" OFF )
if( GRPPI_DOXY_ENABLE )
  add_subdirectory(doxy)
endif( GRPPI_DOXY_ENABLE )

# Installation
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION "include/${CMAKE_PROJECT_NAME}"
        FILES_MATCHING PATTERN "*.h")

# Example applications
option( GRPPI_EXAMPLE_APPLICATIONS_ENABLE "Sample tests" ON )
if( GRPPI_EXAMPLE_APPLICATIONS_ENABLE )
  add_subdirectory(samples)
endif( GRPPI_EXAMPLE_APPLICATIONS_ENABLE )

# Example applications
option( GRPPI_PERFORMANCE_TEST_ENABLE "Performance tests" OFF )
if( GRPPI_PERFORMANCE_TEST_ENABLE )
  add_subdirectory(performance_tests)
endif( GRPPI_PERFORMANCE_TEST_ENABLE )

# Unit Tests
enable_testing()
option( GRPPI_UNIT_TEST_ENABLE "Unit tests" ON )
if( GRPPI_UNIT_TEST_ENABLE )
  add_subdirectory(unit_tests)
endif( GRPPI_UNIT_TEST_ENABLE )
