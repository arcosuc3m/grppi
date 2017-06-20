# Map pattern

The **map** pattern applies an operation to every element in one or multipe data sets, generating a new data set as an output.

The interface to the **map** pattern is provided by function `grppi::map()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi(exec, other_arguments...);
~~~

## Unary map

An unary **Map** takes a data set and transforms each element in the data set by applying an unary function and generating a new data set.

The only interface currently offered for this pattern is based in iterators (following the C++ standard library conventions):

  * The input data set is specified by two iterators.
  * The output data set is specified by an iterator to the start of the output sequence.

---
**Example**
~~~c++
vector<double> v = get_the_vector();
vector<double> w(v.size());
map(exec, begin(v), end(v), begin(w),
    [](double x) { return 2 *x; }
);
~~~
---


## N-ary map

A n-ary **Map** taks multiple data sets and transforms tuple of elements from those data sets by applying a n-ary function and generating a new data set.

The only interface currently offered for this pattern is based in iterators:

  * The first data set is specified by two iterators.
  * The output data set is specified by an iterator to the start of the output sequence.
  * All the other input data sets are specified by iterators to the start of the input data sequences.

---
**Example**
~~~c++
vector<double> v1 = get_first_vector();
vector<double> v2 = get_second_vector();
vector<double> v3 = get_third_vector();
map(exec, begin(v1), end(v1), begin(w),
  [](double x, double y, double z) { return x+y+z; },
  begin(v2), begin(v3)
);
~~~
---