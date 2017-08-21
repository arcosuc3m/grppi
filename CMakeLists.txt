cmake_minimum_required (VERSION 3.2.0)
set(PROJECT_NAME_STR grppi)
project(${PROJECT_NAME_STR})

# Path for additional CMake modules
set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules)

include(${CMAKE_MODULE_PATH}/build-external.cmake)

# Print processor and compiler 
message( STATUS "Detected processor is " ${CMAKE_SYSTEM_PROCESSOR} )
message( STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )
message( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )
message( STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID} )

# Set specific options depending on compiler
if ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" )

    if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 3.4.0))
        message( FATAL_ERROR "Clang version " ${CMAKE_CXX_COMPILER_VERSION} " not supported. Upgrade to 3.4 or above.")
    else()
        message( STATUS "C++ Compiler is Clang" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 " )
    endif()

elseif ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" )

    if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 6.0))
        message( FATAL_ERROR "g++ version " ${CMAKE_CXX_COMPILER_VERSION} " not supported. Upgrade to 6.0 or above.")
    else()
        message( STATUS "C++ Compiler is GNU")
        add_definitions( -std=c++14 )
    endif()

elseif ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
    
    if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 16.0))
        message( FATAL_ERROR "Intel version " ${CMAKE_CXX_COMPILER_VERSION} " not supported. Upgrade to 16.0 or above.")
    else()
        message( STATUS "C++ Compiler is Intel" )
        message( WARNING "WARNING: GrPPI has not been fully tested with Intel compiler." )
    endif()

else()

    message( FATAL_ERROR "Unsupported compiler: " ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

endif()

# GrPPI library
include_directories("${CMAKE_SOURCE_DIR}/include")

# TBB Library
find_package( TBB ) 
if( TBB_FOUND )
    option( GRPPI_TBB_ENABLE "Require TBB library" ON )
else( TBB_FOUND )
    option( GRPPI_TBB_ENABLE "Require TBB library" OFF )
endif( TBB_FOUND )

message( STATUS "TBB: " ${GRPPI_TBB_ENABLE} )
if( GRPPI_TBB_ENABLE )
    include_directories(${TBB_INCLUDE_DIRS} )
    add_definitions( "-DGRPPI_TBB" )
endif( GRPPI_TBB_ENABLE )

# Threads Library
find_package(Threads REQUIRED)

# OMP library
option( GRPPI_OMP_ENABLE "Require OpenMP library" ON )
message( STATUS "OpenMP: " ${GRPPI_OMP_ENABLE})
if( GRPPI_OMP_ENABLE )
    find_package(OpenMP REQUIRED)
    add_definitions( "-DGRPPI_OMP" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif( GRPPI_OMP_ENABLE )

# Fastflow - to be managed
option( GRPPI_FF_ENABLE "Require FastFlow framework" OFF )
message( STATUS "FastFlow: " ${GRPPI_FF_ENABLE} )
if( GRPPI_FF_ENABLE )
    include_directories( ${PROJECT_SOURCE_DIR}/ff )
    # file ( APPEND ${HEADER_FILE} "\n#define FF_ENABLE ON")
    add_definitions( "-DGRPPI_FF" )
    set( FFPATH ${PROJECT_SOURCE_DIR}/ff )
    message( STATUS "FastFlow Path is: " ${FFPATH} )
endif( GRPPI_FF_ENABLE )

# Doxygen Documentation
option( GRPPI_DOXY_ENABLE "Require Doxygen library" OFF )
if( GRPPI_DOXY_ENABLE )
    add_subdirectory(doxy)
endif( GRPPI_DOXY_ENABLE )

# Installation
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION "include/${CMAKE_PROJECT_NAME}"
        FILES_MATCHING PATTERN "*.h")
install( DIRECTORY ${PROJECT_SOURCE_DIR}/ff DESTINATION "${CMAKE_PROJECT_NAME}" )

# Example applications
option( GRPPI_EXAMPLE_APPLICATIONS_ENABLE "Sample tests" ON )
if( GRPPI_EXAMPLE_APPLICATIONS_ENABLE )
    add_subdirectory(samples)
endif( GRPPI_EXAMPLE_APPLICATIONS_ENABLE )

# Unit Tests
enable_testing()
option( GRPPI_UNIT_TEST_ENABLE "Unit tests" OFF )
if( GRPPI_UNIT_TEST_ENABLE )
    add_subdirectory(unit_tests)
endif( GRPPI_UNIT_TEST_ENABLE )

