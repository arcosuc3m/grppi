**daxpy**

This program computes the *daxpy* BLAS operation. That is, given two vectors, `x`
and `y` of size **n** and a coefficient `a`, it computes the operation:

~~~
y = a * x + y
~~~

This program performs the following steps:

1. Generate two vectors with random numbers following an uniform random distribution
between -100.0 and 100.0.
2. Generate a coefficient following an uniform random distribution between 1.0
and 10.0.
3. Compute the daxpy operation.
4. Print the resulting vector.
