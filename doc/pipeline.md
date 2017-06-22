# Pipeline pattern

The **pipeline** pattern receives a data stream that is processed in several stages. Each stage manages the data produced by the previous stage and delivers its result to the next stage.

The interface to the **pipeline** pattern is provided by function `grppi::pipeline()`. As all functions in *GrPPI*, this function takes as its first argument an execution policy.

~~~c++
grppi::pipeline(exec, other_arguments...);
~~~

## Pipeline

The pipeline has an intial phase where the data is initially generated. The next stages will process the data and finally it will arrive to the last stage.

The data in each stage is processed in parallel but there are dependencies between the stages, so a data element cannot be processed by stage 2 if it has not finish stage 1.

---
**Example**
~~~c++
    std::vector<int> output;
    p.ordering=true;
    int a = initial_value;
    pipeline( exec,
        [&]() { 
            a--; 
            if (a == 0) 
                return optional<int>(); 
            else 
                return optional<int>(a); 
        },
        [&]( int k ) {
            return k*2;
        },
        [&]( int x ) {
            output.push_back(x);
        }
    );
~~~
---
