# Farm pattern

The **farm** pattern applies an operation to every element of a data stream, generating a new data stream as output.

The interface to the **farm** pattern is provided by function `grppi::farm()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi(exec, other_arguments...);
~~~

## Farm

The farm receives a data stream that is managed by the generation function that sends each data element to the operation function where they are processed.

Each element of the stream shall be independent from eachother to be processed in parallel.

---
**Example**
~~~c++
int a = initial_value;
std::atomic<int> output;
farm(exec,[&]() {
            a--; 
            if ( a == 0 ) 
                return optional<int>(); 
            else
                return optional<int>( a );
        },
        [&]( int x ) {
            output += x;
        }
    );
~~~
---
