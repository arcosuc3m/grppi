# Stream reduction pattern

The **stream reduction** pattern consists on the use of the reduce pattern over a stream of data. Each element is processed using a reduce operator, that generates a new output element. The output elements are sent to an output stream.

The interface to the **stream reduction** pattern is provided by function `grppi::stream_reduce()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi::stream_readuce(exec, other_arguments...);
~~~

## Stream reduction variants

TODO

## Key elements in stream reduction

TODO

## Details on stream reduction variants

TODO