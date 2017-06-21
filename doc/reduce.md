# Reduce pattern

The **reduce** pattern applies an operation to every element in one data set, generating a new output element. The operation has to be composed of a reduce operator, that, by definition, can be computed in parallel.

The interface to the **reduce** pattern is provided by function `grppi::reduce()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi(exec, other_arguments...);
~~~

## Reduce

An **Reduce** takes a data set and transforms each element in the data set by applying a reduce operation.

The only interface currently offered for this pattern is based in iterators (following the C++ standard library conventions):

  * The input data set is specified by two iterators.
  * The output data set is specified by an iterator to the start of the output sequence.

Depending on the logic of the program the output data set can be simplified to a single variable.

---
**Example**
~~~c++
vector<double> v = get_the_vector();
double out=1;
reduce(p, begin(v), end(v), out, std::divides<double>());
~~~
---
