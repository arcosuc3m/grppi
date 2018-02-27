# GrPPI: Generic Reusable Parallel Patterns Interface

GrPPI is a C++14 library that aims to offer a simplified interface for common
parallel patterns with different implementation back-ends.

Currently, the following back ends are supported:

* Sequential execution.
* OpenMP.
* Intel TBB.
* Native implementation (using ISO C++ threads).
* FastFlow (since version 0.3.1).

The API documentation can be found here:

* Version [0.2](0.2/index.html)
* Version [0.3.1](0.3.1/index.html)
