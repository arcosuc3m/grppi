# Install Notes

## Building GrPPI

### Building the library

GrPPI is a header only library and no build process is required for the library
itself.

However, GrPPI includes CMake scripts for supporting the following actions:

* Building the sample programs.
* Building the unit tests.
* Performing coverage analysis on unit tests.
* Generating the **doxygen** based documentation.
* Installing the library in your system

To setup the build scripts we recommend that you create an out of source
directory under the root GrPPI directory:

~~~
mkdir build
cd build
~~~

Then, you may generate the scripts by just doing:

~~~
cmake ..
make
~~~

**Important Note:** Be sure to invoke make once before modifying you `CMakeCache.txt`
file. This will allow CMake to compile and setup dependent libraries (e.g.
GoogleTest).

### Building the unit tests

To build the unit tests, you need to set configuration variable
`GRPPI_UNIT_TEST_ENABLE` to `ON`. You can do so be using the CMake GUI or by
typing:

~~~
cmake .. -DGRPPI_UNIT_TEST_ENABLE=ON
~~~



### Building the sample programs



## Compilers ##

For using GrPPI you need a C++14 compliant compiler.

GrPPI has been tested with the following compilers:

  * **g++** 6.1. 
  * **clang++** 3.4.

## Required Libraries ##

Miminal support of GrPPI requires the following libraries.

  * [BOOST](http://www.boost.org/) version 1.58 or above.

## Additional Libraries ##

In order to use the **Threading Building Blocks** (TBB) back-end you need to
install the following library:

  * [TBB](https://www.threadingbuildingblocks.org/)

## Unit tests and coverage analysis

For unit testing GrPPI uses the GoogleTest framework. However you do not need to
install yourself. The framework is locally downloaded and compiled in your build
tree to ensure that the right version is used. For more details see section
(building).

If you want to run unit tests and perform coverage analysis you will need:

  * [lcov](https://github.com/linux-test-project/lcov)
    To generate gcov HTML reports.

## Installation ##

1. Create a **build** folder in the project directory

2. Change directory to the build folder and execute **cmake ..**

	2.2. To choose a installation destination use:

   		 cmake -DCMAKE_INSTALL_PREFIX=path/to/folder

3. To install execute **make install**. The dafault folder is /usr/local on UNIX and c:/Program Files on Windows.

### Advanced Options

The project make use of different libraries such as OpenMP (OMP) or Threading Building Blocks (TBB). This libraries can be disabled in the cmake configuration file with ccmake. 

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
