# Map pattern {#mappattern}

The **map** pattern applies an operation to every element in one or multipe data sets, generating a new data set as an output.

The interface to the **map** pattern is provided by function `grppi::map()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi::map(exec, other_arguments...);
~~~

## Map variants

There are several variants:

* *Unary map*: A map taking a single input sequence.
* **N-ary map*: A map taking multiple input sequences.

## Key elements in a map

The key element in a **map** pattern is the **Transformer** operation. The transformar may be a **UnaryTransformer** or a **MultiTransformer**.

A **UnaryTransformer** is any C++ callable entity that takes a data item and transforms it. The input type and the output type may differ. Thus a unary transformer `op` is any operation that, given an input `x` of type `T` and output type `U`, makes valid the following:

~~~c++
U res = op(x);
~~~

A **MultiTransformer** is any C++ callable entity that takes data items, one of each input sequence, and transforms them into an output value. The input types and the output type may differ. Thes a multi-transformer `op` is any operation that, given inputs `x1, x2, ... , xN` of types `T1, T2, ... , TN`and an output type `U`, makes valid the following:

~~~c++
U res = op(x1,x2,...,xN)
~~~

## Details on map variants

### Unary map

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


### N-ary map

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
vector<double> w(v1.size());
map(exec, begin(v1), end(v1), begin(w),
  [](double x, double y, double z) { return x+y+z; },
  begin(v2), begin(v3)
);
~~~
---
