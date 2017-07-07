# Pipeline pattern

The **pipeline** pattern receives a data stream that is processed in several
stages. Each stage processes the data produced by the previous stage and delivers
its result to the next stage.

The interface to the **pipeline** pattern is provided by function
`grppi::pipeline()`. As all functions in *GrPPI*, this function takes as its
first argument an execution policy.

~~~{.cpp}
grppi::pipeline(exec, other_arguments...);
~~~

## Pipeline variants

There are several variants:

* *Standalone pipeline*: Is a top level pipeline. Invoking the algorithm runs the
pipeline.
* *Composable pipeline*: Builds a pipeline object that can be later inserted
into another pattern.

## Key elements in pipeline

The key elements in a **pipeline** are the **Generator** producing data items
and the **Transformer** stages.

A **Generator** is any C++ callable entity that takes zero arguments and
produces data items from a given type. Thus, a **Generator** `gen` is any
operation that, given an output type `U`, makes valid the following:

~~~{.cpp}
U res = gen();
~~~

A **Transformer** is any C++ callable entity that takes a data item and
transforms it. The input type and the output type may differ. Thus, a 
transformer `op` is any operation that, given an input `x` of type `T` and output type
`U`, makes valid the following:

~~~{.cpp}
U res = op(x)
~~~

## Details on pipeline variants


### Standalone pipeline

A *standalone pipeline* generates data from a source and passes the output to
the first stage that applies a transformation to each data item. The resulting
items are passed to the next stage and so on.

Consequently, a pipeline with a **Generator** `gen` and `N` **Transformer**
*stages* (`s1, s2, ..., sN`) performs the following computation:

~~~
sN(...s2(s1(gen())))
~~~

**Note**: Each stage may run concurrently with other stages. However, there are
dependencies between stages, so that every item passes sequentially across
stages.

---
**Example**
~~~{.cpp}
    pipeline(exec,
      [&input]() -> optional<int> {
        int n;
        input >> n;
        if (!input) return {};
        else return n;
      },
      [](int x) { return x*x; },
      [](int x) { return 1.0/x; }.
      [&output]() {
        output << x;
      }
    );
~~~
---

### Composable pipeline

TODO
