# Stencil pattern

The **stencil** pattern applies a transformation to every element in one or
multiple data sets, generating a new data set as an output. The transformation
also takes as an input a neighborhood of the transformed data item.

The interface to the **stencil** pattern is provided by function
`grppi::stencil()`. As all functions in *GrPPI*, this function takes as its
first argument an execution policy.

~~~{.cpp}
grppi::stencil(exec, other_arguments...);
~~~

## Stencil variants

There is a single variant:

* *Unary stencil*: A stencil taking a single input sequence.

## Key elements in stencil

The key elements in a **stencil** are: a **StencilTransformer** operation and a
**NeighbourMapping** operation.

The **StencilTransformer** is any C++ callable entity that takes a data item and
transforms it. Additionally it takes a number of neighbours. The output type may
differ. Thus, a **StencilTransformer** is any 


## Details on stencil variants

TODO

## Stencil

The **Stencil** takes a data set and extracts a set of neighbors elements that are transformed by applying an unary function and generating a new data set.

The only interface currently offered for this pattern is based in iterators (following the C++ standard library conventions):

  * The input data set is specified by two iterators.
  * The output data set is specified by an iterator to the start of the output sequence.

