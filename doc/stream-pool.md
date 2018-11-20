# Stream pool pattern

The **stream pool** pattern receives an initial population whose individuals
represent solution candidates. This pattern manages the population as a stream,
evolving and filtering these individuals until a termination condition is met.

The interface of the **stream pool** pattern is provided by function 
`grppi::stream_pool()`. As all functions in *GrPPI*, this function takes as
its first argument an execution policy.

~~~{.cpp}
grppi::stream_pool(exec, other_arguments...);
~~~

## Key elements in stream pool

The key elements in a **stream pool** are the **Population** and
the **Selection**, **Evolution**, **Filtering** and **Termination** operations.

A **Population** is a container object that represents a set of different 
**individuals** representing potential solutions to the problem. This container
object should provide at least the member function `size()

The **Selection** is any callable C++ entity that receives a population as input
and selects a subset of individuals from the population. This selection process
may modify the population without side effects (e.g. removing a selected individual).

The **Evolution** is any callable C++ entity that receives the selected 
individuals and returns a set of modified individuals.

The **Filtering** is any callable C++ entity that receives the subset of selected 
individuals and the evolved individuals of an iteration and returns an individual.
The resulting individual is eventually introduced in the initial population.

The **Termination** is any callable C++ entity that receives the filtered individual
and returns a value contextually convertible to bool. 




