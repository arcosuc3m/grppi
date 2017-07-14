# Map/reduce pattern

The **map/reduce** pattern combines a **map** and **reduce** operation in a single pattern. Firstly one or more input data sets are *mapped* by applying a transformation operation. Then, the results are combined by means of a reduction operation.

The interface to the **mapreduce** pattern is provided by function `grppi::map_reduce()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~{.cpp}
grppi::map_reduce(exec, other_arguments...);
~~~

**Note**: A **map/reduce** could be also expressed by the composition of a **map** and a **reduce**. However, **map/reduce** fuses both stages allowing for extra optimizations.

## Map/reduce variants

There are several variants:

* Unary map/reduce: Applies a *map/reduce* to a single data set.
* N-ary map/reduce: Applies a *map/reduce* taking multiple data sets that are combined during the map stage.

## Key elements in a map/reduce

There are two central elements of a **map/reduce**: the **Transformation** used for the *map* stage, and the **Combination** used for the *reduce* stage.

The **Transformation** is an operation that takes one element from each input data set and generates one value of the intermediate data set. The number and type of arguments of that operation needs to match the type and number of input data sets.

The **Combination** is a binary operation taking two values of the type of the intermediate data set and combining them into a value of the same type.

## Details on map/reduce variants

### Unary map_reduce

An unary **map/reduce** takes a single data set and performs consecutively the **map** and the **reduce** stages, returning the reduced value.

The only interface currently offered for this pattern is based in iterators (following the C++ standard library conventions):

  * The input data set is specified by two iterators.

---
**Example**
~~~{.cpp}
auto res = map(exec,
  begin(v), end(v),
  [](string s) { return stod(s); },
  [](double x, double y) { return x+y; }
);
~~~
---


### N-ary map_reduce

A n-ary **map/reduce** takes multiple data sets and performs consecutively the **map** and **reduce** stage, returning the reduced value.

The only interface currently offered for this pattern is based in iterators (following the C++ standard library conventions):

  * The first data set is specified by two iterators.
  * All the other input data sets are specified by iterators to the start of the input data sequences, assuming that the size of all sequences are at least as large as the first sequence.

---
**Example**
~~~{.cpp}
auto res = map(exec,
  begin(v), end(v),
  [](double x, double y) { return x*y; },
  [](double x, double y) { return x+y; },
  begin(w)
);
~~~
---