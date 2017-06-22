# Stencil pattern

The **stencil** pattern applies an operation to a set of neighbor elements in one data set, generating a new data set as an output.

The interface to the **map** pattern is provided by function `grppi::stencil()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi::stencil(exec, other_arguments...);
~~~

## Stencil

The **Stencil** takes a data set and extracts a set of neighbors elements that are transformed by applying an unary function and generating a new data set.

The only interface currently offered for this pattern is based in iterators (following the C++ standard library conventions):

  * The input data set is specified by two iterators.
  * The output data set is specified by an iterator to the start of the output sequence.

