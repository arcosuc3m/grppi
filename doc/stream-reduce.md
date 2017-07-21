# Stream reduction pattern

The **stream reduction** pattern consists on the use of the reduce pattern over
a stream of data. Each element is processed using a reduce operator, that generates 
a new output element. The output elements are sent to an output stream.

The interface to the **stream reduction** pattern is provided by function 
`grppi::stream_reduce()`. As all functions in *GrPPI*, this function takes as 
its first argument an execution policy.

~~~{.cpp}
grppi::stream_readuce(exec, other_arguments...);
~~~

## Stream reduction variants

TODO

## Key elements in stream reduction

The key element in a **stream reduction** is the **Combiner** operation. 

A **Combiner** is any C++ callable entity, that is able to combine two values
into a single value.
A **Combiner** `cmb` is any operation taking two values `x` and
`y` of types `T` and `U` and returning a combined value of type `T`, making valid
the following:

~~~{.cpp}
T x;
U y;
T res = cmb(x,y);
~~~

## Details on stream reduction variants

TODO
