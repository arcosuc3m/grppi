# Install Notes

## Building GrPPI

### Building the library

GrPPI is a header only library and no build process is required for the library
itself.

However, GrPPI includes CMake scripts for supporting the following actions:

* Building the unit tests.
* Performing coverage analysis on unit tests.
* Generating the **doxygen** based documentation.
* Building the sample programs.
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

Then, you can build the unit tests by typing:

~~~
make
~~~

### Disabling specific GrPPI backends

You may want to disable specific back-ends. GrPPI offers specific variables to
control this issue:

* `GRPPI_OMP_ENABLE`: Enable/disable OpenMP backend.
* `GRPPI_TBB_ENABLE`: Enable/disable TBB backend.

### Running the unit tests

To run all the unit tests you can do:

~~~
make test
~~~

or alternatively:

~~~
ctest
~~~

### Performing coverage analysis

To performe a coverage analysis type:

~~~
make coverage
~~~

The coverage HTML reports are generated under `unit_tests/mycov/index.html`.

### Documentation generation

Documentation generatio is disabled by default. However, if you wish to build
the documentation yourself, you may enable the option:

~~~
cmake .. -DGRPPI_DOXY_ENABLE=ON
make
~~~

This will generate a doc directory under your build tree with the generated
documentation.

**Note:** You will need a **doxygen** in your system to make use of this option.
You will also need **graphviz**.


### Building the sample programs

GrPPI includes a number of example programs under directory **samples**. To
build all samples you may use:

~~~
cmake .. -DGRPPI_EXAMPLE_PROGRAMS_ENABLE=ON
make
~~~

### Installing GrPPI

If you want to install GrPPI in your system you can select to install in the
default directory:

~~~
sudo make install
~~~

This will install the header files under `/usr/local/include/grppi`

You can specify a different install directory to CMake:

~~~
cmake .. -DCMAKE_INSTALL_PREFIX=path/to/folder
make install
~~~

## Supported Compilers ##

For using GrPPI you need a C++14 compliant compiler.

GrPPI has been tested with the following compilers:

  * **g++** 6.1. 
  * **clang++** 3.4.

If you want to use a different compiler than the default one, you can specify it
by doing:

~~~
cmake .. -DCMAKE_CXX_COMPILER=clang++
make
~~~

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

