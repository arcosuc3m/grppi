# Map_reduce pattern

The **map_reduce** pattern applies an operation to every element in one or multipe data sets, generating a new data set as an output. Then it applies a reduce operation to the elements of the data set.

The interface to the **mapreduce** pattern is provided by function `grppi::map_reduce()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi(exec, other_arguments...);
~~~

## Unary map_reduce

An unary **Map_reduce** takes a data set and transforms each element in the data set ino a smaller data set. This subset receives the an unary function and generating a new data subset. Then a reduce function is applied to each subset generating a new element of the output set.

The only interface currently offered for this pattern is based in iterators (following the C++ standard library conventions):

  * The input data set is specified by two iterators.
  * The output data set is specified by an iterator to the start of the output sequence.

---
**Example**
~~~c++
std::vector<std::vector<int>> mat(3);
for(int i=0;i<mat.size();i++) {
    mat[i] = std::vector<int> (3);
    std::iota(begin(mat[i]), end(mat[i]), 0);
} 

std::vector<int> out(3);
int aux;
map_reduce(p, begin(mat), end(mat), begin(out),
       [&](auto & in, auto &aux){ 
              std::vector<int> mult(in.size()); 
              for(auto col = 0; col!= in.size(); col++){
                   mult[col] = in[col] * 2;    
              }     
              return mult;  
       },
       std::plus<int> ()
       ,aux
);

~~~
---


## N-ary map_reduce

A n-ary **Map_reduce** taks multiple data sets and transforms tuple of elements from those data sets by applying a n-ary function and generating a new data subset. Each subset is processed by the reduction function and generates a new set.

The only interface currently offered for this pattern is based in iterators:

  * The first data set is specified by two iterators.
  * The output data set is specified by an iterator to the start of the output sequence.
  * All the other input data sets are specified by iterators to the start of the input data sequences.

---
**Example**
~~~c++
std::vector<std::vector<int>> mat(3);
for(int i=0;i<mat.size();i++) {
    mat[i] = std::vector<int> (3);
    std::iota(begin(mat[i]), end(mat[i]), 0);
} 

std::vector<int> v(3);
for( int i= 0 ; i< v.size(); i++){
     v[i] = 2;
}
std::vector<int> v2(3);
for( int i= 0 ; i< v2.size(); i++){
     v2[i] = 1;
}
std::vector<int> out(3);

map_reduce(p, begin(mat), end(mat), begin(out),
       [&](auto & in, auto & v, auto &v2){ 
              std::vector<int> mult(in.size()); 
              for(auto col = 0; col!= in.size(); col++){
                   mult[col] = in[col] * v[col] + v2[col];    
              }     
              return mult;  
       },
       std::plus<int> (),
       v,
       v2
);
~~~
---