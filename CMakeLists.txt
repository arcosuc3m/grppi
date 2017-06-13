#
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take
# their definition and dump it in the cache along with proper
# documentation, otherwise set  CMAKE_BUILD_TYPE to Debug prior to
# calling PROJECT() 
#

# IF(DEFINED CMAKE_BUILD_TYPE)
#    SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
# build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
# Release RelWithDebInfo MinSizeRel.")
#    MESSAGE(STATUS "Build Type from command line " ${CMAKE_BUILD_TYPE})
# ELSE()
#    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
# options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
# RelWithDebInfo MinSizeRel.")
#    MESSAGE(STATUS "Default Build Type: Release  (change with
#    -DCMAKE_BUILD_TYPE=Debug | Release | RelWithDebInfo | MinSizeRel")
# ENDIF()

 
cmake_minimum_required (VERSION 2.8.11)
project( grppi )


# Define a path for modules before accesing default. Needed to find TBB
#set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} )
# Modified because cmake.modules doesnt exist in the project directory
set( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules ${PROJECT_SOURCE_DIR} )

# PROBLEM: THIS WAY OF SELECTING THE COMPILER IS NOT RECOMMENDED
#  ERRORS MAY OCCUR IF CMAKE WAS USED BEFORE WITH OTHER COMPILER 
#   --> remove build directory and create again
# Solution 1: CXX=/PATH_TO/clang++ cmake ..
#  Source: https://cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F

# Solution 2: Is possible to set the environement variables CC and CXX before
#  executing cmake, so that values are taken as compiler
#  export CXX=/PATH_TO/g++ EXAMPLE: export CXX=/usr/bin/g++ and after cmake ..
#  export CXX=/usr/bin/g++
#  export CXX=/opt/clang/bin/clang++

# Change manually or with ccmake to select the compiler route
#option (COMPILER_PATH "Path to compiler" ON)
#if(COMPILER_PATH)
#	set(CMAKE_CXX_COMPILER "g++")
#else()
#	set(CMAKE_CXX_COMPILER "/opt/local/bin/g++-mp-6")
#endif()
#set(CMAKE_C_COMPILER "gcc")


# INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
# INCLUDE (${CMAKE_ROOT}/Modules/CheckTypeSize.cmake)
# INCLUDE (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
# INCLUDE (${CMAKE_ROOT}/Modules/CheckCXXSourceCompiles.cmake)
# INCLUDE (${CMAKE_ROOT}/Modules/TestBigEndian.cmake)
# INCLUDE (${CMAKE_ROOT}/Modules/CheckSymbolExists.cmake)


# Print processor and compiler 
message( STATUS "Detected processor is " ${CMAKE_SYSTEM_PROCESSOR} )
message( STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )
message( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )
message( STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID} )

# Set specific options depending on compiler
if ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" )
    message( STATUS "C++ Compiler is Clang" )
    set( PPICM_HAS_CLANGXX "TRUE" )
   # set( CMAKE_CXX_FLAGS "-stdlib=libc++ -std=gnu++11" )
    set( CMAKE_CXX_FLAGS "-std=c++14" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register -Wno-empty-body -Wno-unused-value" )
   # add_definitions( -stdlib=libc++ -std=gnu++11 )  # Or -std=c++11
elseif ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" )
    message( STATUS "C++ Compiler is GNU")
    execute_process( COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GXX_VERSION )
    if(NOT(GXX_VERSION VERSION_GREATER 4.1))
        set( GCC_TOO_OLD 1 )
        message( FATAL_ERROR "g++ Version < 4.1, some targets will be skipped" )
        message( WARNING "Use CC/CXX env variable to define a newer compiler, e.g. export CC=/usr/bin/gcc-4.2; export CXX=/usr/bin/g++-4.2" )
    else(NOT(GXX_VERSION VERSION_GREATER 4.1))
        add_definitions( -std=c++14 ) # or -std=c++0x
	    add_definitions( -O3 -g )
        # FASTFLOW SPECIFIC	
	    add_definitions(-DNO_DEFAULT_MAPPING) # -DBLOCKING_MODE 
        #add_definitions( -std=c++11 ) # or -std=c++0x 
    endif(NOT(GXX_VERSION VERSION_GREATER 4.1))

elseif ( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
    message( STATUS "C++ Compiler is Intel" )
    message( WARNING "Compliance check not fully implemented yet" )
endif()

include_directories("${CMAKE_SOURCE_DIR}/include")

# BOOST library
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

set(Boost_LIBRARIES system thread)
find_package(Boost 1.36.0 REQUIRED ${Boost_LIBRARIES})

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


# Create header file with the XX_ENABLE flags
#set (HEADER_FILE "include/enable_flags.hpp")
#file (WRITE ${HEADER_FILE} "//Enable Flags")

# TBB Library 
option( TBB_ENABLE "Require TBB library" ON )
message( STATUS "TBB: " ${TBB_ENABLE} )
if( TBB_ENABLE )
    include_directories(${TBB_INCLUDE_DIRS} )
    find_package(TBB REQUIRED)
    add_definitions( "-DGRPPI_TBB" )
#    file ( APPEND ${HEADER_FILE} "\n#define TBB_ENABLE ON")
endif( TBB_ENABLE )

# Threads Library
find_package(Threads REQUIRED)

# OMP library
option( OMP_ENABLE "Require OpenMP library" ON )
message( STATUS "OpenMP: " ${OMP_ENABLE})
if( OMP_ENABLE )
    find_package(OpenMP REQUIRED)
    add_definitions( "-DGRPPI_OMP" )
#    file ( APPEND ${HEADER_FILE} "\n#define OMP_ENABLE ON")
endif( OMP_ENABLE )

# CUDA library
option( CUDA_ENABLE "Require CUDA library" OFF )
message( STATUS "CUDA: " ${CUDA_ENABLE})
if( CUDA_ENABLE )
    
    if ( APPLE )
        message(WARNING "Apple NVCC no longer supported with gcc")
    else() 
        find_package(CUDA)

        if (CUDA_VERSION VERSION_GREATER "7.4")

            list(APPEND nvcc_flags -ccbin "${CUDA_HOST_COMPILER}")

            # clang++ needs libomp to compile CUDA
            # sudo apt-get install libomp-dev

            # set( PPIHEADERS_GPU
            #     ${PPI}/map_thrust.hpp
            #     ${PPI}/farm_thrust.hpp
            #     ${PPI}/reduce_thrust.hpp
            #     ${PPI}/pipeline_thrust.hpp
            #     ${PPI}/mapreduce_thrust.hpp
            #     ${PPI}/stream_reduce_thrust.hpp
            # )

            list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda; -std=c++11; -Wno-deprecated-gpu-targets")
            

        endif()
    endif( APPLE )
endif( CUDA_ENABLE )

# Doxygen Documentation
add_subdirectory(doxy)


# Installation
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION "include/${CMAKE_PROJECT_NAME}"
        FILES_MATCHING PATTERN "*.h")

# Test creation. Related with CMakeLists.txt on tests folder.
option( BUILD_TESTS "Build tests" ON )
message( STATUS "Configuring >>tests<< directory" )
if( BUILD_TESTS )
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(unit_tests)
    add_subdirectory(legacy_utest)
endif( BUILD_TESTS )
