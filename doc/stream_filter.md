# Stream Filter pattern

The **Stream_filter** pattern receives data for an input stream that is discarded or selected according to a filter function. It generates an output stream with the filtered elements.

The interface to the **stream_filter** pattern is provided by function `grppi::stream_filter()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi::stream_filter(exec, other_arguments...);
~~~
