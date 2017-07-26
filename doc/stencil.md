# Stencil pattern

The **stencil** pattern applies a transformation to every element in one or
multiple data sets, generating a new data set as an output. The transformation
also takes as an input a neighborhood of the transformed data item.

The interface to the **stencil** pattern is provided by function
`grppi::stencil()`. As all functions in *GrPPI*, this function takes as its
first argument an execution policy.

~~~{.cpp}
grppi::stencil(exec, other_arguments...);
~~~

## Stencil variants

There are several variants:

* *Unary stencil*: A stencil taking a single input sequence.
* *N-ary stencil*: A stencil taking multiple input sequences.

## Key elements in stencil

The key elements in a **stencil** are: a **StencilTransformer** operation and a
**Neighbourhood** operation.

A **Neighbourhood** operation is any C++ callable entity that takes one or more 
iterators to data items and generates a neighbourhood. Depending on 
the number of iterators taking it may be a **UnaryNeighbourhood** or a 
**MultiNeighbourhood**.

Consequently, a **UnaryNeighbourhood** is any operation `op` that given an iterator 
value `it` and an output neighbourhood type `N` makes valid the following:

~~~{.cpp}
N n = op(it);
~~~

A **MultiNeighbourhood** is any operation `op` that given **n** iterator values
`it1`, `it2`, ... , `itN` and an output neighbourhood type `N` makes valid the following:

~~~{.cpp}
N n = op(it1, it2, ..., itN);
~~~

The **StencilTransformer** is any C++ callable entity that takes 
one iterator to a data item and the result of a **Neighbourhood** operation and performs
a transformation. 

Thus, a **StencilTransfomer** is any operatoio `op`that give an iterator value `it`and 
the result of a **Neighbourhood** operation `n` makes valid the following:

~~~{.cpp}
R r = op(it, n);
~~~

## Details on stencil variants

### Unary stencil

An unary **stencil** takes a data set and transforms each element in the data set by
applying a transformation to each data item using the data item and its neighbourhood.

The only interface currently offered for this pattern is based in iterators
(following the C++ standard library conventions):

* The input data set is specified by two iterators.
* The output data set is specified by an iterator to the start of the output sequence.

---
**Example**: Stencil operation adding neighbours in a vector.
~~~{.cpp}
vector<double> v = get_the_vector();
vector<double> w(v.size());
grppi::stencil(ex, begin(v), end(v), begin(w),
    // Add current element and neighbours
    [](auto it, auto n) {
      return *it + accumulate(begin(n), end(n)); 
    }
    // Neighbours are prev and next
    [&](auto it) {
      vector<double> r;
      if (it!=begin(v)) r.push_back(*prev(it));
      if (distance(it,end(end))>1) r.push_back(*next(it));
      return r;
    }
~~~
---

### N-ary stencil

An n-ary **stencil** takes multiple data sets and transforms each element in the data set by
applying a transformation to each data item form the first data set and the neighbourhood
obtained from all data sets.

The only interface currently offered for this pattern is based in iterators:

* The first data set is specified by two iterators.
* The output data set is specified by an iterator to the start of the output sequence.
* All the other input data sets are specified by iterators to the start of the input data sequences.

---
**Example**: Stencil operation adding neighbours in two vector.
~~~{.cpp}
vector<double> v1 = get_the_vector();
vector<double> v2 = get_the_vector();
vector<double> w(v.size());

auto get_around = [](auto i, auto b, auto e) {
};

grppi::stencil(ex, begin(v), end(v), begin(w),
    // Add current element and neighbours
    [](auto it, auto n) {
      return *it + accumulate(begin(n), end(n)); 
    }
    // Neighbours are prev and next
    [&](auto it1, auto it2) {
      vector<double> r;
      if (it1!=begin(v1)) r.push_back(*prev(it1));
      if (distance(it1,end(v1))>1) r.push_back(*next(it1));
      if (it2!=begin(v2)) r.push_back(*prev(it2));
      if (distance(it2,end(v2))>2) r.push_back(*next(it2));
      return r;
    }
);
~~~
---
