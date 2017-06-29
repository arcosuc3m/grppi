# Reduce pattern

The **reduce** pattern is a data pattern that combines all the values in a data set using a binary combination operation.

The interface to the **reduce** pattern is provided by function `grppi::reduce()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~{.cpp}
grppi::reduce(exec, other_arguments...);
~~~

## Reduction variants

There is a single variant of the reduction:

* **Sequence reduction without initial value**: Reduces a sequence of values
with no initial value.

* **Sequence reduction with initial value**: Reduces a sequence of values with
an initial value.


## Key elements in a reduction

The key element of a reduction is the **Combiner** operation. The **Combiner**
may be an **HeterogeneousCombiner** or an **HomogeneousCombiner**.

A **Combiner** is any C++ callable entity, that is able to combine two values
into a single value. The requirements for the **Combiner** are different
depending on whether the reduction uses an initial value
(**HeterogeneousCombiner**) or not (**HomogeneousCombiner**).

An **HomogeneousCombiner** `cmb` is any operatingo taking two values `x`and `y`
of type `T` and returning a combined value of type `T`, making valid the
following.

~~~{.cpp}
T x, y;
T res = cmb(x,y);
~~~

An **HeterogeneousCombiner** `cmb` is any operations taking two values `x` and
`y` of types `T` and `U` and returning a combined value of type `T`, making valid
the following:

~~~{.cpp}
T x;
U y;
T res = cmb(x,y);
~~~

## Details on reduction variants

### Sequence reducction without identity

This kind of sequence reduction performs a reduction of a sequence of values
`x1, x2, ..., xN` by combining them with an **HomogeneousCombiner** `cmb`. The
combinations assume that `cmb` is *associative*, but not commutative.
Consequently, different associative orders may be used:

* `cmb(cmb(cmb(cmb(x1,x2),x3), ...), xN)`
* `cmb(x1,cmb(x2,cmb(x3,...cmb(xN-1,xN))))`
* `cmb(cmb(cmb(x1,x2),cmb(x3,x4)),cmb(...))`
* ...

The only interface currently offered for this pattern is based in iterators
(following the C++ standard library conventions):

* The input data set is specified by two iterators.
* The result of the reduction is returned.

---
**Example**: Add values in a sequence of integers
~~~{.cpp}
vector<int> v = get_the_vector();
auto sum = reduce(exec, begin(v), end(v), 
  [](int x, int y) { return x+y; }
);
~~~

**Note**: Reducing without identity value an empty sequence is undefined.

### Sequence reduction with identity

This kind of sequence reduction performs a reduction of a sequence of values
`x1, x2, ..., xN` by combining them with an **HeterogeneousCombiner** `cmb` that
has an identity value `id`. The first argument of a combination may be either
the identity value or the result of another combination.
The combinations assume that `cmb` is *associative*, but not commutative.
Consequently, different associative orders may be used:

* `cmb(cmb(cmb(cmb(cmb(id,x1),x2),x3), ...),xN)`
* `cmb(cmb(cmb(cmb(id,x1),x2),cmb(cmb(id,x3),x4)),...)`
* ...

The only interface currently offered for this pattern is based in iterators
(following the C++ standard library conventions):

* The input data set is specified by two iterators.
* The identity value is provided as an input value.
* The result of the reduction is returned.

---
**Example**: Add the lenghts of a sequence of strings.
~~~{.cpp}
vector<string> v = get_the_vector();
auto result = reduce(exec,
  begin(v), end(ve), 0
  [](int n, string s) { return n + s.length(); }
);
~~~
---

**Note**: Reducing with identity value an empty sequence has a result the
identity value.

