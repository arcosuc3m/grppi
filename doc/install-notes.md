# Install Notes

## Building the GrPPI library

GrPPI is a *header-only* library. Consequently, no build process is required for
the library itself.

However, GrPPI includes CMake scripts for supporting the following actions:

* Building the unit tests.
* Performing coverage analysis on unit tests.
* Generating the **doxygen** based documentation.
* Building the sample programs.
* Installing the library in your system

To setup the build scripts we recommend that you create an *out-of-source*
directory under the GrPPI root directory:

~~~
mkdir build
cd build
~~~

Then, you may generate the scripts by just doing:

~~~
cmake ..
~~~

Finally, you launch the build process with:

~~~
make
~~~

## Building the unit tests

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

## Control of back-ends

GrPPI offers multiple back-ends:

* *Sequential*: Always enabled.
* *Native*: Always enabled.
* *OpenMP*: Can be enabled/disabled.
* *Intel TBB*: Can be enabled/disabled.
* *FastFlow*: Can be enabled/disabled.

You may want to disable specific back ends. GrPPI offers specific variables to
control this issue:

* `GRPPI_OMP_ENABLE`: Enable/disable OpenMP back-end.
* `GRPPI_TBB_ENABLE`: Enable/disable TBB back-end.
* `GRPPI_FF_ENABLE`: Entable/disable FastFlow back-end.

### OpenMP back-end

The OpenMP back-end is controlled by CMake option `GRPPI_OMP_ENABLE` (values can
be `ON` or `OFF`). You can enable/disable this back-end with this option.

GrPPI uses CMake's `FindOpenMP` to detect if your compiler supports OpenMP. In
case your compiler supports OpenMP `GRPPI_OMP_ENABLE` will default to `ON`. If
your compiler does not support OpenMP any attempt to enable GrPPI/OpenMP
back-end will be ignored.

If you have and OpenMP compliant compiler and GrPPI refuses to enable the OpenMP
back-end, please, open an issue providing details.

### Intel TBB back-end

The Intel TBB back-end is controlled by CMake option `GRPPI_TBB_ENABLE` (values
can be `ON` or `OFF`). You can enable/disable this back-end with this option.

GrPPI tries to detect if your system has Intel TBB installed. In case, Intel TBB
is found `GRPPI_TBB_ENABLE` will default to `ON`. If Intel TBB is not found,
any attempt to enable GrPPI/TBB back-end will be ignored.

Please refer to (Additional Libraries)[#additional-libraries] for details on
Intel TBB install.

If you have installed Intel TBB and GrPPI refuses to enable the TBB back-end,
please, open an issue providing details.

### FastFlow back-end

The FastFlow back-end is controlled by CMake option `GRPPI_FF_ENABLE` (values
can be `ON` or `OFF`). You can enable/disable this back-end with this option.

GrPPI tries to detect if your system has FastFlow installed. In case, FastFlow
is found `GRPPI_TBB_ENABLE` will default to `ON`. If FastFlow is not found,
any attempt to enable GrPPI/FF back-end will be ignored.

Please refer to (Additional Libraries)[#additional-libraries] for details on
Intel TBB install.

If you have installed FastFlow and GrPPI refuses to enable the FastFlow back-end,
please, open an issue providing details.

## Running the unit tests

To run all the unit tests you can do:

~~~
make test
~~~

or alternatively:

~~~
ctest
~~~

### Performing coverage analysis

To perform a coverage analysis type:

~~~
make coverage
~~~

The coverage HTML reports are generated under `unit_tests/mycov/index.html`.

### Documentation generation

Documentation generation is disabled by default. However, if you wish to build
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
cmake .. -DGRPPI_EXAMPLE_APPLICATIONS_ENABLE=ON
make
~~~

### Installing GrPPI

If you want to install GrPPI in your system, it will be installed using as base
path the directory defined by `CMAKE_INSTALL_PREFIX`. You do not need to do
anything special if you want to install in the default location.

~~~
sudo make install
~~~

On many Linux systems, this will install the header files under
`/usr/local/include/grppi`.

You can specify a different install directory to CMake:

~~~
cmake .. -DCMAKE_INSTALL_PREFIX=path/to/folder
make install
~~~

## Supported Compilers ##

For using GrPPI you need a C++14 compliant compiler.

GrPPI has been tested with the following compilers:

  * **g++** 6.3, 7.2. 
  * **clang++** 3.9, 4.0, 5.0.

If you want to use a different compiler than the default one, you can specify it
by doing:

~~~
cmake .. -DCMAKE_CXX_COMPILER=clang++-4.0
make
~~~

## Required Libraries ##

No external library is strictly required for basic GrPPI use. It only hard
dependency is the C++ standard library.


## Additional Libraries ##

In order to use the **Intel Threading Building Blocks** (TBB) back-end you need
to install the following library:

  * [TBB](https://www.threadingbuildingblocks.org/)

In order to use the **FastFlow** back-end you need to install it. You can obtain
a recent version from:

  * [FastFlow](https://github.com/fastflow/fastflow)

## Unit tests and coverage analysis

For unit testing GrPPI uses the GoogleTest framework. 

You can obtain Get it from:

  * [GoogleTest](https://github.com/google/googletest)

If you want to run unit tests and perform coverage analysis you will need:

  * [lcov](https://github.com/linux-test-project/lcov)
    To generate gcov HTML reports.

