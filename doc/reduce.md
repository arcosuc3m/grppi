# Reduce pattern

The **reduce** pattern is a data pattern that combines all the values in a data set using a binary combination operation.

The interface to the **reduce** pattern is provided by function `grppi::reduce()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi(exec, other_arguments...);
~~~

## Reduction variants

There is a single variant of the reduction:

* Sequence reduction: Takes a sequence represented by two iterators and reduces the values.


## Key elements in a reduction

The key element of a reduction is the combination **Operation**. The operation may be any C++ callable entity. This operation, is a binary operation taking two values of the same type and combining them into a value of the same type.

## Details on reduction variants

### Sequence reduction

The only interface currently offered for this pattern is based in iterators (following the C++ standard library conventions):

  * The input data set is specified by two iterators.

The reduction result is returned by the pattern interface.

---
**Example**
~~~c++
vector<double> v = get_the_vector();
auto result = reduce(exec,
  begin(v), end(ve),
  [](double x, double y) { return x+y; }
);
~~~
---
