# Stream Reduce pattern

The **stream_reduce** pattern consists on the use of the reduce pattern over a stream of data. Each element is processed using a reduce operator, that generates a new output element. The output elements are sent to an output stream.

The interface to the **stream_reduce** pattern is provided by function `grppi::stream_reduce()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi(exec, other_arguments...);
~~~
