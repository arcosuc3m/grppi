# Divide/conquer pattern

The **divide/conquer** pattern splits a problem into two or more independent
subproblems until a base case is reached. The base case is solved directly and
the results of the subproblems are combined until the final solution of the
original problem is obtained.

The interface to the **Divide&Conquer** pattern is provided by function
`grppi::divide_conquer()`. As all functions in *GrPPI*, this function takes as
its first argument an execution policy.

~~~{.cpp}
grppi::divide_conquer(exec, other_arguments...);
~~~

## Divide/conquer variants

There is a single variant:

* **Generic problem  divide/conquer**: Applies the *divide/conquer* pattern to a
  generic problem and returns a solution.

## Key elements in divide/conquer

The key elements of the **divide/conquer** pattern are: a **Divider** operation that
divides a problem into subproblems, a **Predicate** that signals if a problem is already
an elemental problem, a **Solver** operation that is used to solve
a subproblem, and a **Combiner** operation that is used to merge results of
subproblems.

A **Divider** is any C++ callable entity that takes a problem and returns a
collection of subproblems. The returned collection of subproblems must be
iterable. This allows returning any standard C++ sequence container, or even a
plain array. When a problem cannot be divided into subproblems, the divider
returns a collection with a single subproblem.

A **Predicate** is any C++ callable entity that takes a problem and returns a
boolean value. Returning *true* means that the problem is already elemental and
should be solved using the **Solver**. Returning *false* means that the problem 
needs to be divided into subproblems by the **Divider**.

The **Solver** is any C++ callable entity that takes a problem and turns it into
a solution. The signature of the solver takes as argument a problem and returns
the corresponding solution.

The **Combiner** is any C++ callable entity capable to combine two solutions.
The signature of the combiner takes two solutions and returns a new combined solution.

## Details on divide/conquer variants

### Generic divide/conquer

The **divide/conquer** pattern takes an input problem and generates an output problem.

---
**Example**: Merge sort of an array.
~~~{.cpp}
vector<int> v{1,3,5,7,2,4,6,8};

struct range {
  std::vector<int>::iterator first, last;
  auto size() const { return distance(first,last); }
};

std::vector<range> divide(range r) {
  auto mid = r.first + distance(r.first,r.last)/2;
  return { {r.first,mid} , {mid, r.last} };
}

range problem{begin(v), end(v)};

auto res = grppi::divide_conquer(exec,
  problem,
  [](auto r) -> vector<range> { return divide(r); },
  [](auto r) { return r.size()<=1; },
  [](auto x) { return x; },
  [](auto r1, auto r2) {
    std::inplace_merge(r1.first, r1.last, r2.last);
    return range{r1.first, r2.last};
  }
);
~~~
---

