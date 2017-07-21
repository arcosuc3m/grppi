# Map/reduce pattern

The **map/reduce** pattern combines a **map** and **reduce** operation in a
single pattern. In a **map/reduce** operation, firstly one or more input data
sets are *mapped* by applying a transformation operation. Then, the results are
combined by means of a reduction operation.

The interface to the **map/reduce** pattern is provided by function
`grppi::map_reduce()`. As all functions in *GrPPI*, this function takes as its
first argument an execution policy.

~~~{.cpp}
grppi::map_reduce(exec, other_arguments...);
~~~

**Note**: A **map/reduce** could be also expressed by the composition of a
**map** and a **reduce**. However, **map/reduce** may fuse both stages,
allowing for extra optimizations.

## Map/reduce variants

There are several variants:

* *Unary map/reduce*: A **map/reduce** taking a single input sequence.
* *N-ary map/reduce*: A **map/reduce** taking multiple input sequences that are
combined during the *map* stage.

## Key elements in a map/reduce

There are two central elements of a **map/reduce**: the **Transformer** used for
the *map* stage, and the **Combiner** used for the *reduce* stage.

A **Transformer** may be a **UnaryTransformer** or a **MultiTransformer**.

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
that, given the inputs `x1, x2, ... , xN` of types `T1, T2, ... , TN` and an output
type `U`, makes valid the following:

~~~{.cpp}
U res = op(x1, x2, ..., xN);
~~~

A **Combiner** is any C++ callable entity, that is able to combine two values
into a single value. A Combiner `cmb` is any operation taking two values `x` and
`y` of types `T` and `U` and returning a combined value of type `T`, making valid the
following:

~~~{.cpp}
T x;
U y;
T res = cmb(x,y);
~~~

## Details on map/reduce variants

### Unary map_reduce

An unary **map/reduce** takes a single data set and performs consecutively the
**map** and the **reduce** stages, returning the reduced value.

The only interface currently offered for this pattern is based on iterators
(following the C++ standard library conventions):

  * The input data set is specified by two iterators.

A unary **map/reduce** also requires an identity value for the **Combiner**.

---
**Example**: Transforms a sequence of strings to its corresponding double values
and computes the addition of those values.
~~~{.cpp}
vector<string> v { "1.0", "2.0", "3.5", "0.25" };
auto res = grppi::map_reduce(exec,
  begin(v), end(v),
  0.0,
  [](string s) { return stod(s); },
  [](double x, double y) { return x+y; }
);
// res == 6.75
~~~
---


### N-ary map/reduce

A n-ary **map/reduce** takes multiple data sets and performs consecutively the
**map** and **reduce** stages, returning the reduced value.

The only interface currently offered for this pattern is based on iterators
(following the C++ standard library conventions):

  * The first data set is specified by two iterators.
  * All the other input data sets are specified by iterators to the start of the
    input data sequences, assuming that the size of all sequences are, at least,
    as large as the first sequence.

A n-ary **map/reduce** also requires an identity value for the **Combiner**.

---
**Example**: Compute scalar vector between two vectors of doubles.
~~~{.cpp}
v = get_first_vector();
w = get_second_vector();
auto res = grppi::map_reduce(exec,
  begin(v), end(v),
  0.0,
  [](double x, double y) { return x*y; },
  [](double x, double y) { return x+y; },
  begin(w)
);
~~~
---


## Additional examples of **map/reduce**

---
**Example**: Count word appearances in text lines.
~~~{.cpp}
vector<string> lines{"Far far away, behind the word mountains, far from the ",
                     "countries Vokalia and Consonantia, there live the blind ",
                     "texts. Separated they live in Bookmarksgrove right at ",
                     "the coast of the Semantics, a large language ocean."};
map<string,int> init;

auto res = grppi::map_reduce(e,
  lines.begin(),
  lines.end(),
  init,
  [](string & l){
    // Split lines in substrings represeting words
    istringstream line{l};
    vector<string> words{istream_iterator<string>{line},
                         istream_iterator<string>{}};
    std::map<string,int> word_count;
    // Initialize map with the line words
    for (auto & w : words) { word_count[w]++; }
    return word_count;
  },
  [](auto partial_count, auto word_count){
    // Compute partial results
    for (auto & w : word_count) {
      partial_count[w.first]+= w.second;
    }
    return partial_count;
  }
);
// res = {{"the", 5}, {"far", 2}, {"live", 2}, ... }
~~~
---
