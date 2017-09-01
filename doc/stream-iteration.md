# Stream iteration pattern

The **stream iteration** pattern allows loops in data stream processing. An
operation is applied to a data item until a predicate is satisfied. When the
predicate is met, the result is sent to the output stream.  This streaming
pattern can only be used inside another pattern and consequently does not take
an execution policy itself, but uses the execution policy of its enclosing
pattern.

The interface to the **stream iteration** pattern is provided by function
`grppi::repeat_until()`. 

~~~{.cpp}
grppi::pipeline(ex,
  stage1,
  grppi::repeat_until(arguments...),
  stage2,
  stage3,
  ...);
~~~

## Stream iteration variants

There is a single variant:

* *Composable stream iteration*: Defines a stream iteration that can be used as a
building block by another pattern (e.g. pipeline).

## Key elements in stream iteration

The key elements in a **stream iteration** are a **Transformer** used to transform data items, 
and a **Predicate** that defines when the iteration should finish.

The **Transformer** may be any C++ callable entity that takes a data item and applies
a transformation to it. Thus, a **Transformer** `op` is any operation that, given an input
value `x` of type `T`makes valid the following:

~~~{.cpp}
U res{transformer(x)};
~~~

The **Predicate** may be any C++ callable entity that takes  a data item
and returns a value that is contextually convertible to `bool`.
Thus, a predicate `pred` is any operation, that given a
value `x` of type `T`, makes the following valid:

~~~{.cpp}
do { /*...*/ } while (!predicate(item));
~~~

## Details on stream iteration variants

### Stand-alone stream iteration 

A composable **stream iteration** has two elements:
 
* A **Transformer** of values.
* A **Predicate** for terminating the iteration.

---
**Example**: For every natural number x, print the first value x*2^n
that is greater than 1024.
~~~
grppi::pipeline(ex
  [i=0,max=100]() mutable -> optional<int> {
    if (i<max) return i++;
    else return {};
  },
  grppi::repeat_until(
    [](int x) { return 2*x; },
    [](int x) { return x>1024; }),
  [](int x) { cout << x << endl; }
);
~~~
---

For composing complex patterns, the `repeat_until()` function may be used to
create an object that may be used later in the composition.

---
**Example**: For every natural number x, print the first value x*2^n
that is greater than 1024.
~~~
auto loop = grppi::repeat_until(
    [](int x) { return 2*x; },
    [](int x) { return x>1024; });

grppi::pipeline(ex
  [i=0,max=100]() mutable -> optional<int> {
    if (i<max) return i++;
    else return {};
  },
  loop,
  [](int x) { cout << x << endl; }
);
~~~
---
