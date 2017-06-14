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
        message( FATAL_ERROR "Clang version outdated. Upgrade to 3.4 or above.")
    else()
        message( STATUS "C++ Compiler is Clang" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 " )
    endif()

elseif ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" )

    if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 6.0))
        message( FATAL_ERROR "g++ version outdated. Upgrade to 6.0 or above.")
    else()
        message( STATUS "C++ Compiler is GNU")
        add_definitions( -std=c++14 -fno-delete-null-pointer-checks )
    endif()

elseif ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
    
    if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 16.0))
        message( FATAL_ERROR "Intel version outdated. Upgrade to 16.0 or above.")
    else()
        message( STATUS "C++ Compiler is Intel" )
        message( WARNING "Compliance check not fully implemented yet" )
    endif()
    
endif()

# GrPPI library
include_directories("${CMAKE_SOURCE_DIR}/include")

# BOOST library
set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME ON) 

set(Boost_LIBRARIES system thread)
find_package(Boost 1.5 REQUIRED ${Boost_LIBRARIES})

if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	set( CMAKE_BOOST_FLAGS "-lboost_system -lboost_thread" )
	set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_BOOST_FLAGS}" )
	# Windows
	if (WIN32)
  		# disable autolinking in boost
  		add_definitions( -DBOOST_ALL_NO_LIB )
		# maybe not needed
		add_definitions( -DBOOST_ALL_DYN_LINK )
	endif()
else()
	message( FATAL_ERROR "Boost library is not installed in your system" )    
endif()

# TBB Library 
option( GRPPI_TBB_ENABLE "Require TBB library" ON )
message( STATUS "TBB: " ${GRPPI_TBB_ENABLE} )
if( GRPPI_TBB_ENABLE )
    include_directories(${TBB_INCLUDE_DIRS} )
    find_package(TBB REQUIRED)
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

# CUDA library
option( GRPPI_CUDA_ENABLE "Require CUDA library" OFF )
message( STATUS "CUDA: " ${CUDA_ENABLE})
if( GRPPI_CUDA_ENABLE )
    
    if ( APPLE  )

        message(WARNING "Apple NVCC no longer supported with gcc")
    else() 
        find_package(CUDA)
        if (CUDA_VERSION VERSION_GREATER "7.4")
            list(APPEND nvcc_flags -ccbin "${CUDA_HOST_COMPILER}")
            list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda; -std=c++11; -Wno-deprecated-gpu-targets")
        endif()
    endif( APPLE )
endif( GRPPI_CUDA_ENABLE )

# Doxygen Documentation
add_subdirectory(doxy)

# Installation
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION "include/${CMAKE_PROJECT_NAME}"
        FILES_MATCHING PATTERN "*.h")

# Tests
enable_testing()
option( GRPPI_TESTS_ENABLE "Regular tests" OFF )
if( GRPPI_TESTS_ENABLE )
    add_subdirectory(tests)
endif( GRPPI_TESTS_ENABLE )

# Sample Tests
option( GRPPI_EXAMPLE_APPLICATIONS_ENABLE "Sample tests" OFF )
if( GRPPI_EXAMPLE_APPLICATIONS_ENABLE )
    add_subdirectory(samples)
endif( GRPPI_EXAMPLE_APPLICATIONS_ENABLE )

# Unit Tests
option( GRPPI_UNIT_TEST_ENABLE "Unit tests" OFF )
if( GRPPI_UNIT_TEST_ENABLE )
    add_subdirectory(unit_tests)
endif( GRPPI_UNIT_TEST_ENABLE )

# Legacy Tests
option( GRPPI_LEGACY_UTEST_ENABLE "Legacy unit tests" OFF )
if( GRPPI_LEGACY_UTEST_ENABLE )
    add_subdirectory(legacy_utest)
endif( GRPPI_LEGACY_UTEST_ENABLE )
