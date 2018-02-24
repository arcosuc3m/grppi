# Context

The **context** pattern (or *execution context*) is a streaming pattern that
allows to run a subset of a pipeline with a different execution policy than the
one used by the containing structure.  This streaming pattern can only be used
inside another pattern.

The interface to the **context** pattern is provided by function
`grppi::run_with()`.

~~~{.cpp}
grppi::pipeline(exec1,
  stage1,
  run_with(exec2, 
    pipeline(stage2, stage3)
  ),
  stage4,
  stage5);
~~~

## Context variants

There is a single variant:

* Composable *context*: Defines a context that can be used as a building block by
another pattern (e.g. a *pipeline*).

## Key elements in a contex

The key elements of a **context** are the **Execution** and the **Transformer**.

The **execution** is the *execution policy* to be applied inside the new
context.

The central element in a context is the **Transformer**. The operation may be
any C++ callable entity, including another composable streaming pattern.  Thus,
a transformer `op` is any operation that, given an input value `x` of type `T`
makes valid the following:

~~~{.cpp}
U res{transformer(x)};
~~~

## Valid context compositions

When two execution policies are composed some simplifications are applied.
Besides, that some cases are not fully optimized in the current version.

### Outer sequential policy

Given any execution policy `ex`:

~~~{.cpp}
grppi::pipeline(seq,
  stage1,
  grppi::run_with(ex, farm(4, stage2)),
  stage3);
~~~

It does not matter which is the inner execution policy as items are generated
one bye one and no actual concurrency is possible.

Consequently, the example is equivalente to:

~~~{.cpp}
grppi::pipeline(seq,
  stage1,
  grppi::run_with(seq, farm(4, stage2)),
  stage3);
~~~

### Outer native policy

TBD

### Outer OpenMP policy

TBD

### Outer TBB policy

TBD

### Outer FastFlow policy

TBD

## Details on context variants

### Composable context

A *composable context* applies a **Transformer** to each data item in a stream.
The context can be inserted as a stage into another upper level pattern (which
will be responsible for generation and consumption)

---
**Example**: Use a context as a stage of a composed pipeline.
~~~{.cpp}
grppi::pipeline(exec,
  stageA,
  stageB,
  grppi::run_with(exec2, [](auto x) {
    return x.length();
  }),
  stageC
);
~~~
---
**Note**: For brevity we do not show here the details of other stages.

