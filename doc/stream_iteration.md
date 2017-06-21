# Stream Iteration pattern

The **stream_iteration** pattern evaluates a condition over the elements of a data stream. While the condition is true, an operation is performed over a data element. When the condition becomes false, the element is send to the output stream.

The interface to the **stream_iteration** pattern is provided by function `grppi::stream_iteration()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi(exec, other_arguments...);
~~~
