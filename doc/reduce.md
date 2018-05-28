# Reduce pattern

The **reduce** pattern is a data pattern that combines all the values in a data set using a binary combination operation.

The interface to the **reduce** pattern is provided by function `grppi::reduce()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~{.cpp}
grppi::reduce(exec, other_arguments...);
~~~

## Reduction variants

There is a single variant of the reduction:

* **Sequence reduction without identity value**: Reduces a sequence of values
with an identity value.

## Key elements in a reduction

The key element of a reduction is the **Combiner** operation. 

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

## Details on reduction variants

### Sequence reduction with identity

Performs a reduction of a sequence of values
`x1, x2, ..., xN` by combining them with a **Combiner** `cmb` that
has an identity value `id`. The first argument of a combination may be either
the identity value or the result of another combination.
The combinations assume that `cmb` is *associative*, but not commutative.
Consequently, different associative orders may be used:

* `cmb(cmb(cmb(cmb(cmb(id,x1),x2),x3), ...),xN)`
* `cmb(cmb(cmb(cmb(id,x1),x2),cmb(cmb(id,x3),x4)),...)`
* ...

There are two interfaces for the reduction with identity:

  * A *range* based interface.
  * An *iterator* based interface.

#### Range based interface

The range based interface specifies sequences as ranges.
A **range** is any type satisfying the `grppi::range_concept`.
In particular, any STL sequence container is a **range**.
Thus, the only sequence provided to `grppi::reduce` is:

  * The input data set is provided by a range:

---
**Example**: Add the numbers in a sequence.
~~~{.cpp}
vector<long> v = get_the_values();
auto result = reduce(exec,
  v, 0L,
  [](long x, long y) { return x+y; }
);
~~~
---

**Note**: Reducing with identity value an empty sequence has a result the
identity value.

#### Iterator based interface

The iterator based interface specifies sequences in terms of iterators
(following the C++ standard library conventions):

* The input data set is specified by two iterators.
* The identity value is provided as an input value.
* The result of the reduction is returned.

---
**Example**: Add the lenghts of a sequence of strings.
~~~{.cpp}
vector<long> v = get_the_values();
auto result = reduce(exec,
  begin(v), end(ve), 0L,
  [](long x, long y) { return x+y; }
);
~~~
---

**Note**: Reducing with identity value an empty sequence has a result the
identity value.

