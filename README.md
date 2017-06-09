# Generic Reusable Parallel Pattern Interface - GrPPI

## Introduction ##

**GrPPI** is an open source *generic and reusable parallel pattern programming interface* developed at Univ. Carlos III of Madrid. Basically, **GrPPI** accommodates a layer between developers and existing parallel programming frameworks targeted to multi-core processors, such as ISO C++ Threads, OpenMP and Intel TBB, and accelerators. It also support GPU programming by means of CUDA Thrust. To achieve this goal, the interface leverages modern C++ features, metaprogramming concepts, and template-based programming to act as switch between these frameworks. Furthermore, its compact design facilitates the development of parallel applications, hiding away the complexity behind the use of concurrency mechanisms. The parallel patterns supported by GrPPI are targeted for both stream processing and data-intensive applications and can be composed among them to match more complex constructions. In a nutshell, **GrPPI** advocates for a usable, simple, generic, and high-level parallel pattern interface, allowing users to implement parallel applications without having a deep understanding of today's parallel programming frameworks and third-party interfaces.

Currently, **GrPPI** supports the following patterns:

  * Divide-and-conquer

  * Farm

  * Map

  * Map-reduce

  * Pipeline

  * Reduce

  * Stencil

  * Stream-filter

  * Stream-reduce

## Install and compile instructions

See the [install and compile notes](INSTALL.md).

## Publications describing GrPPI

### Overview publication

Please cite this publication in any work using our library:

* **A Generic Parallel Pattern Interface for Stream and Data Processing**. David del Rio, Manuel F. Dolz, Javier Fernández, J. Daniel García. *Concurrency and Computation: Practice and Experience*. ISSN: 1532-0634. DOI: [10.1002/cpe.4175](http://dx.doi.org/10.1002/cpe.4175).


### Other references

* **A C++ Generic Parallel Pattern Interface for Stream Processing**. David Del Río Astorga, Manuel F. Dolz, Luis Miguel Sanchez, Javier Garcia Blas and J. Daniel Garcia. *16th International Conference on Algorithms and Architectures for Parallel Processing (ICA3PP)*. Granada, Spain. 14-16/12/2016. pp. 74-84. DOI: [10.1007/978-3-319-49583-5_5](http://dx.doi.org/10.1007/978-3-319-49583-5_5)

