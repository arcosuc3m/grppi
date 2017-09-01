# Farm pattern

The **farm** pattern (or *task-farm*) is a streaming pattern that applies an
operation to every element in a stream. This streaming pattern can only be used
inside another pattern and consequently does not take an execution policy
itself, but uses the execution policy of its enclosing pattern.

The interface to the **farm** pattern is provided by function `grppi::farm()`.

~~~{.cpp}
grppi::pipeline(exec,
  stage1,
  grppi::farm(arguments...),
  stage3,
  stage4);
~~~

## Farm variants

There is a single variant:

* Composable *farm*: Defines a farm that can be used as a building block by
another pattern (e.g. a *pipeline*).

## Key elements in a farm

The key elements of a **farm** are the **cardinality** and the **Transformer**.

The **cardinality** is the number of replicas of the farm that can be
concurrently executed.

The central element in a farm is the **Transformer**. The operation may be any
C++ callable entity. This operation, is a unary operation taking a data item and
returning its transformation. Thus, a transformer `op` is any operation that,
given an input value `x` of type `T` makes valid the following:

~~~{.cpp}
U res{transformer(x)};
~~~

## Details on farm variants

### Composable farm

A *composable farm* applies a **Transformer** to each data item in a stream. The
**farm** can be inserted into another upper level pattern (which will
be responsible for generation and consumption)

---
**Example**: Use a farm as a stage of a composed pipeline.
~~~{.cpp}
grppi::pipeline(exec,
  stageA,
  stageB,
  grppi::farm(4, [](auto x) {
    return x.length();
  }),
  stageC
);
~~~
---
**Note**: For brevity we do not show here the details of other stages.

For composing complex patterns, the `farm()` function may be used to create an object that may be used later in the composition.

---
**Example**: Build a farm as a composable pattern and use later as a stage of a
pipeline.
~~~{.cpp}
auto print_long_words = grppi::farm(3, [](auto x) {
  if (x.length() > 4) std::cout << x << std::endl;
});

grppi::pipeline(exec,
  stageA,
  stageB,
  print_long_words,
  stageC
);
~~~
---
**Note**: For brevity we do not show here the details of other stages.
