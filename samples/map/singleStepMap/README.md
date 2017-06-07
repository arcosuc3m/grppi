**arithmetic_add**

This example shows the functionality of the Map pattern with one step or phase.

Two vectors are created with the size given by the user.
Then the elements of the vector are filled using its index as value.

The parallelized task consist on multiplying each value by two and storing the value in the output vector.

To compile the program: 
g++ main.cpp -I ../../../include/ -std=c++14 -lboost_system -lboost_thread -lpthread -ltbb -fopenmp -DGRPPI_OMP -DGRPPI_TBB
