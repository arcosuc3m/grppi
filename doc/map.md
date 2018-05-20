# Map pattern 

The **map** pattern applies an operation to every element in one or multiple
data sets, generating a new data set as an output.

The interface to the **map** pattern is provided by function `grppi::map()`. As
all functions in *GrPPI*, this function takes as its first argument an execution
policy.

~~~{.cpp}
grppi::map(exec, other_arguments...);
~~~

## Map variants

There are several variants:

* *Unary map*: A map taking a single input sequence.
* *N-ary map*: A map taking multiple input sequences.

## Key elements in a map

The key element in a **map** pattern is the **Transformer** operation. 

The transformer may be a **UnaryTransformer** or a **MultiTransformer**.

A **UnaryTransformer** is any C++ callable entity that takes a data item and
transforms it. The input type and the output type may differ. Thus, a unary
transformer `op` is any operation that, given an input `x` of type `T` and
output type `U`, makes valid the following:

~~~{.cpp}
U res = op(x);
~~~

A **MultiTransformer** is any C++ callable entity that takes data items, one of
each input sequence, and transforms them into an output value. The input types
and the output type may differ. Thus, a multi-transformer `op` is any operation
that, given inputs `x1, x2, ... , xN` of types `T1, T2, ... , TN` and an output
type `U`, makes valid the following:

~~~{.cpp}
U res = op(x1,x2,...,xN)
~~~

## Details on map variants

### Unary map

An unary **map** takes a data set and transforms each element in the data set by
applying an unary function and generating a new data set.

There are two interfaces for the unary map:

  * A *range* based interface.
  * An *iterator* based interface.

#### Range based interface

The *range based* iterface specifies sequences as ranges. A **range** is any type
satisfying the `grppi::range_concept`. In particular, any STL container is
a **range**. Thus, sequences provided to `grppi::map` are:

  * The input data set is specified by a range.
  * The output data set is specified by a range.

---
**Example**: Doubling values in a vector with ranges.
~~~{.cpp}
vector<double> v = get_the_vector();
vector<double> w(v.size());
map(exec, v, w,
    [](double x) { return 2 *x; }
);
~~~
---

#### Iterator based interface

The iterator based interface specifies sequences in terms of iterators
(following the C++ standard library conventions):

  * The input data set is specified by two iterators.
  * The output data set is specified by an iterator to the start of the output sequence.

---
**Example**: Doubling values in a vector with iterators.
~~~{.cpp}
vector<double> v = get_the_vector();
vector<double> w(v.size());
map(exec, begin(v), end(v), begin(w),
    [](double x) { return 2 *x; }
);
~~~
---


### N-ary map

A n-ary **map** takes multiple data sets and transforms a tuple of elements from
those data sets by applying a n-ary function and generating a new data set.

There are two interfaces for the n-ary map:

  * A *range* based interface.
  * An *iterator* based interface.

#### Range based n-ary map

The *range based* iterface specifies sequences as ranges. A **range** is any type
satisfying the `grppi::range_concept`. In particular, any STL container is
a **range**. Thus, sequences provided to `grppi::map` are:

  * The input data sets are specified by ranges packaged into a `grppi::zip_view` by `grppi::zip`.
  * The output data set is specified by a range.

---
**Example**: Computing the addition of three vectors with ranges.
~~~{.cpp}
vector<double> v1 = get_first_vector();
vector<double> v2 = get_second_vector();
vector<double> v3 = get_third_vector();
vector<double> w(v1.size());
map(exec, grppi::zip(v1,v2,v3), w,
  [](double x, double y, double z) { return x+y+z; },
);
~~~
---
#### Iterator based n-ary map

The iterator based interface specifies sequences in terms of iterators
(following the C++ standard library conventions):

  * The input data sets are specified by a tuple of iterators to the start
    of each sequence
  * Additionally, the size of those sequences is specified by either an
    iterator to the end of the first sequence or by an integral value.
  * The output data set is specified by an iterator to the start of the output
    sequence.

---
**Example**: Computing the addition of three vectors.
~~~{.cpp}
vector<double> v1 = get_first_vector();
vector<double> v2 = get_second_vector();
vector<double> v3 = get_third_vector();
vector<double> w(v1.size());
map(exec, std::make_tuple(begin(v1), begin(v2), begin(v3)), end(v1),
  begin(w),
  [](double x, double y, double z) { return x+y+z; },
);
map(exec, std::make_tuple(begin(v1), begin(v2), begin(v3)), v1.size(),
  begin(w),
  [](double x, double y, double z) { return x+y+z; },
);
~~~
---
