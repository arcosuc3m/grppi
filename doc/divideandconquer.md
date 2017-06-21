# Divide&Conquer pattern

The **Divide&Conquer** pattern starts with an initial problem and splits it into two or more subproblems until a base case is reached. The base case is solved directly and the results of the subproblems are merged providing a final solution to the initial problem.

The interface to the **Divide&Conquer** pattern is provided by function `grppi::divide_and_conquer()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi(exec, other_arguments...);
~~~

## Divide&Conquer

The **Divide&Conquer* pattern receives a data set and by means of a divide function, it is break down into subproblems. Each subproblem will follow this logic until the base case is reached, at that point the operation is applied to the subproblem generating a partial solution. Finally, the partial solutions will be joined by the merge function, returning the final result.


