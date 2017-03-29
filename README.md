![logo](https://cormoran.arcos.inf.uc3m.es/rephrase/generic-pattern-interface-with-ff/raw/master/logo.svg)

**Generic Pattern Interface - GrPPI**

## Introduction ##
Generic Pattern Interface (GrPPI) is a tool that works as an intermediate layer between the existing parallel pattern frameworks and developers, providing an interface that simplifies the development of parallel applications by the homogenization of the different libraries syntaxes in a common one. 

The main advantages of this framework are:

  * Help programmers to take advantage of the parallel hardware resources

  * Provide a simple way of porting applications between different parallel programming models and platforms

The parallel patterns implemented are:

  * Divide-and-conquer

  * Farm

  * Map

  * Map-reduce

  * Pipeline

  * Reduce

  * Stencil

  * Stream-filter

  * Stream-reduce

## Full Supported Compilers ##

  * **g++** from GNU. 
  * **clang** from LLVM

  > Note: The versions of the compilers must be compatible with c++14

## Required Libraries ##

  * [BOOST](http://www.boost.org/)

## Additional Libraries ##
  * [FastFlow](http://calvados.di.unipi.it/)
  
  * [CUDA](https://developer.nvidia.com/cuda-downloads)

  * [TBB](https://www.threadingbuildingblocks.org/)

  * [lcov](https://github.com/linux-test-project/lcov)
  	To use coverage with Google Test.

## Additional Programs ##

  * [ccmake](https://cmake.org/cmake/help/v3.0/manual/ccmake.1.html)
  	To enable/disable or modify options of the cmake configuration file through the terminal.

  * [cmake-gui](https://cmake.org/cmake/help/v3.0/manual/cmake-gui.1.html)
    Graphical tool to enable/disable or modify options of the cmake configuration file.

## Installation ##

1. Create a **build** folder in the project directory

2. Change directory to the build folder and execute **cmake ..**

	2.2. To choose a installation destination use:

   		 cmake -DCMAKE_INSTALL_PREFIX=path/to/folder

3. To install execute **make install**. The dafault folder is /usr/local on UNIX and c:/Program Files on Windows.

### Advanced Options

The project make use of different libraries such as OpenMP (OMP), Threading Building Blocks (TBB) or CUDA. This libraries can be disabled in the cmake configuration file with ccmake. 

## Using different compilers ##

1. The compiler must be selected before executing cmake in the build folder. If cmake has been executed and want to change the compiler, delete build folder or its contents.

2. To select the compiler is recommended to change the environement variable CXX, export CXX=/PATH_TO/COMPILER. Once defined the cmake can be executed. For example:

	2.1. GNU:   export CXX=/usr/bin/g++

	2.2. Clang: export CXX=/opt/clang/bin/clang++

3. Is possible to change the CXX variable at the same time the cmake is executed:
	 CXX=/PATH_TO/COMPILER cmake ..


## Using the library ##

To use the library there are several options:

1. Modify the project CMakeLists.txt file in the project directory and include the program files desired.

2. Install the library and include the library in the compilation command. For example:

 2.1. cmake -DCMAKE_INSTALL_PREFIX=/opt/

 2.2. sudo make install
      
 2.3. g++ test.cpp -I /opt/GrPPI -I /opt/GrPPI/fastflow/ -std=c++14 -lboost_system -lboost_thread



## Testing ##
The project includes some tests that can be used to check the proper installation of the GrPPI or can be used as examples.
To use the test:

1. Create a **build** folder in the project directory

2. Change directory to the build folder and execute **cmake ..**

3. Execute **make** to create the test files inside build/test folder

4. Execute **ctest** to execute all the tests

> Note: The tests can be executed manually inside the build/test folder

## Google Tests ##
The tests have been also implemented with Google Test. To create them the Google Test option must be set with ccmake and the recommended compiler is GNU g++. The Google Test tests have the suffix "_GT".

This tests can be generated with the regular tests following the same steps as in "Testing" section.

This tests include also the **coverage** option. Once the cmake has been executed, is possible to start the coverage executing make coverage_test-Name. At the end of the execution a new web browser window will be open with the coverage results. By default the browser selected is Mozilla Firefox. For example:

	make coverage_farm1_GT 

The coverage files will be created in build/tests/mycov.

## CUDA Tests ##
The project includes some CUDA based tests. This tests are not implemented with Google Test. CUDA can be enabled/disabled with cmake, by default is enabled. If CUDA is disabled the corresponding tests will not be compiled.

To compile a CUDA program that uses GrPPI the following line code can be used:

nvcc program-name.cu -I /PATH/GrPPI -I /PATH/GrPPI/fastflow/  --expt-extended-lambda -std=c++11 -Wno-deprecated-gpu-targets -lboost_system -lboost_thread
