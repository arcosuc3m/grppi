## Doxygen
The present directory contains the required files to build a Doxygen documentation of the GrPPI project. 

Doxyfile has the configuration of the documentation generator and CMakeLists.txt contains the CMake directives to build it.

### usage:

1. Create build directory
2. Change directory to build
3. Execute cmake ..
4. Execute make doc_doxygen

The doc directory will be created and the doxygen documentation shall be located inside in two formats, HTMl and Latex.