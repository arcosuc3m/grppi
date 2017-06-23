# Divide/conquer pattern

The **divide/conquer** pattern splits a problem into two or more independent subproblems until a base case is reached. The base case is solved directly and the results of the subproblems are combined until the final solution of the original problem si obtained.

The interface to the **Divide&Conquer** pattern is provided by function `grppi::divide_conquer()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~{.cpp}
grppi::divide_conquer(exec, other_arguments...);
~~~

## Divide/conquer variants

There is a single variant:

* **Generic problem  divide/conquer**: Applies de *divide/conquer* pattern to a generic problem and returns a solution..

## Key elements in divide/conquer

The key elements of **divide/conquer** pattern are: a **Divider** operation that divides a problem into subproblems, a **Solver** operation that is used to solve a subproblem, and a **Combiner** operation that is used to merge results of subproblems.

A **Divider** is any C++ callable entity that takes problem and returns collection of solutions. The signature of the divider takes as argument the problem and returns a collection of suproblems. The returned collection of subproblems must be iterable. This allows returning any standard C++ sequence container, or even a plain array. When a problem cannot be divided into subproblems the divider returns a collection with a single subproblem.

The **Solver** is any C++ callable entity that takes a problem and turns it into a solution. The signature of the solver takes as argument a problem and returns the corresponding solution.

The **Combiner** is any C++ callable entity capable to combine two solutions. The signature of the combiner taks two solutions and returns a new solution.

## Details on divide/conquer variants

### Generic divide/conquer

The **divide/conquer** pattern takes an input problem and generates an output problem.

---
**Example**: Missing example
~~~{.cpp}
vector<int> v{1,3,5,7,2,4,6,8};

struct range {
  vector<int>::iterator first, last;
  auto size() const { return distance(first,last); }
};

vector<range> divide(range r) {
  auto mid = first + distance(first,last)/2;
  return { make_pair(first,mid), mak_pair(mid+1,last) };
}

auto res = grppi::divide_conquer(exec,
  range{begin(v), end(v)},
  [](auto r) -> vector<range> {
    if (r.size()>1) return { r };
    else return divide(r);
  },
  [](auto x) { return x; }
  [](auto r1, auto r2) {
    std::inplace_merge(r1.first, r2.first, r2.last);
    return range{r1.first, r2.last};
  }
);
~~~
---

