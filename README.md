# Generic Reusable Parallel Pattern Interface - GrPPI

[![HitCount](http://hits.dwyl.io/arcosuc3m/grppi.svg)](http://hits.dwyl.io/arcosuc3m/grppi)

## Introduction ##

**GrPPI** is an open source *generic and reusable parallel pattern programming
interface* developed at University Carlos III of Madrid. Basically, **GrPPI**
accommodates a layer between developers and existing parallel programming
frameworks targeted to multi-core processors, such as ISO C++ Threads, OpenMP
, Intel TBB, and FastFlow. To achieve this goal, the interface leverages
modern C++ features, meta-programming concepts, and generic programming
to act as switch between those frameworks. 

Furthermore, its compact design facilitates the development of parallel
applications, hiding away the complexity behind the use of concurrency
mechanisms. The parallel patterns supported by GrPPI are targeted for both
stream processing and data-intensive applications and can be composed among
them to match more complex constructions. In a nutshell, **GrPPI** advocates
for a usable, simple, generic, and high-level parallel pattern interface,
allowing users to implement parallel applications without having a deep
understanding of today's parallel programming frameworks and third-party
interfaces.

Currently, **GrPPI** supports the following patterns:

  * Data parallel patterns
    * [Map](doc/map.md)
    * [Reduce](doc/reduce.md)
    * [Map/Reduce](doc/map-reduce.md)
    * [Stencil](doc/stencil.md)

  * Task parallel patterns
    * [Divide-and-conquer](doc/divide-conquer.md)

  * Streaming patterns
    * [Pipeline](doc/pipeline.md)
    * [Farm](doc/farm.md)
    * [Stream filter](doc/stream-filter.md)
    * [Stream reduction](doc/stream-reduce.md)
    * [Stream iteration](doc/stream-iteration.md)

Additionally, streaming patterns allow the use of [multi-context](doc/context.md) execution,
aiming to allow the combination of multiple back-ends for the execution of
a single pipeline.

## Install and compile instructions

See the [install and compile notes](doc/install-notes.md).

## Publications describing GrPPI

### Overview publication

Please cite this publication in any work using our library:

* **A Generic Parallel Pattern Interface for Stream and Data Processing**. David del Río, Manuel F. Dolz, Javier Fernández, J. Daniel García. *Concurrency and Computation: Practice and Experience*. ISSN: 1532-0634. DOI: [10.1002/cpe.4175](http://dx.doi.org/10.1002/cpe.4175).


### Other references


* **Parallelizing and optimizing LHCb-Kalman for Intel Xeon Phi KNL processors**. Plácido Fernández, David del Río, Manuel .F. Dolz, Javier Fernández, Omar Awile, J. Daniel Garcia *PDP 2018*

* **Supporting Advanced Patterns in GrPPI: a Generic Parallel Pattern Interface**. David del Río, Manuel F. Dolz, Javier Fernández, and J. Daniel Garcia. *Auto-DaSP 2017 (Euro-Par 2017)*. Santiago de Compostela, Spain. 28/8-1/9/2017. pp. 55-67. DOI: [10.1007/978-3-319-75178-8_5](https://doi.org/10.1007/978-3-319-75178-8_5)

* **Finding parallel patterns through static analysis in C++ applications**. David del Río, Manuel F. Dolz, Luís M. Sanchez, J. Daniel Garcia, Marco Danelutto, and Massimo Torquati. *International Journal of High Performance Computing Applications*. 2017. DOI: [10.1177/1094342017695639](https://doi.org/10.1177/1094342017695639)

* **A C++ Generic Parallel Pattern Interface for Stream Processing**. David Del Río, Manuel F. Dolz, Luis Miguel Sanchez, Javier Garcia Blas and J. Daniel Garcia. *16th International Conference on Algorithms and Architectures for Parallel Processing (ICA3PP)*. Granada, Spain. 14-16/12/2016. pp. 74-84. DOI: [10.1007/978-3-319-49583-5_5](http://dx.doi.org/10.1007/978-3-319-49583-5_5)
### Acknowledgments

The **GrPPI** library has been partially supported by:

* Project ICT 644235 **"REPHRASE: REfactoring Parallel Heterogeneous Resource-aware Applications"** funded by the European Commission through H2020 program (2015-2018).

* Project TIN2016-79673-P **“Towards Unification of HPC and Big Data Paradigms”** funded by the Spanish Ministry of Economy and Competitiveness (2016-2019).
