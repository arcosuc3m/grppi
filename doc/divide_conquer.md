# Divide/conquer pattern

The **divide/conquer** pattern splits a problem into two or more independent subproblems until a base case is reached. The base case is solved directly and the results of the subproblems are combined until the final solution of the original problem si obtained.

The interface to the **Divide&Conquer** pattern is provided by function `grppi::divide_conquer()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi::divide_conquer(exec, other_arguments...);
~~~

## Divide/conquer variants

There is a single variant:

* Divide/Conquer:

## Key elements in divide/conquer

The key elements of **divide/conquer** pattern are: a **Divider** operation that divides a problem into subproblems, a **Solver** operation that is used to solve a subproblem, and a **Combiner** operation that is used to merge results of subproblems.

## Details on divide/conquer variants

### Divide/Conquer

The **divide/conquer** pattern takes an input problem and generates an output problem.

---
**Example**: Missing example
~~~c++
~~~
---

