#include <iostream>
#include <vector>
#include <numeric>
#include <experimental/optional>

#include "grppi.h"
#include "task/parallel_execution_task.h"
#include "task/fifo_scheduler.h"
#include "task/simple_task.h"

using namespace grppi;

int main(){

  fifo_scheduler<simple_task> sched;
  parallel_execution_task<fifo_scheduler<simple_task>> p{sched};

  int size = 1000;


  // SrtingFile(filepath, delimiter, read_function)
  // StringFile("fs:filepath", "\n", [](string s){ return atoi(s);}); 
  //                       -> Obtain an iterator per data block.
  //                       -> Each worker gets data block.
  //                       -> On each data block obtain the string and divide it by
  //                          the delimiter and generates a vector.
  //                       -> Transforms the vector using the funcion provided
  // File<Type>("filename");
  //                       -> The type define how the data is stored
  //                       -> File may not work with types of variable size.
  // Limitation : What if a data structure is divided into two different data blocks¿?
  
  std::vector<std::vector<int>> matrix(size,std::vector<int>(size,2));
  std::vector<std::vector<int>> matrixout(size,std::vector<int>(size,0));
  std::vector<long> v(size);
  std::vector<double> v2(size);
  for(int i = 0; i<size;i++) v[i] = i;

  std::cout<<"NESTED MAP"<<std::endl;
  map(p, matrix.begin(), matrix.end(), matrixout.begin(), [&size,&p](std::vector<int> b){
    std::vector<int> aux(size,0);
    map(p, b.begin(), b.end(), aux.begin(), [](int val){ return 2*val;});
    return aux;
  });

  for(int i = 0; i<size;i++) {
     for(int j=0; j < size; j++) 
       std::cout<<matrixout[i][j]<<" ";
     std::cout<<std::endl;
  }

  std::cout<<"SINGLE MAP"<<std::endl;
  map(p, v.begin(), v.end(), v.begin(), [](int b){ return b;});
  for(int i = 0; i<size;i++) std::cout<<v[i] << " ";
  std::cout<<std::endl; 

  std::cout<<"REDUCE"<<std::endl;
  long res1 = reduce(p, v.begin(), v.end(), (long) 0, [](long b, long a){ return b+a;});

  std::cout<<res1<<std::endl; 
  std::cout<<"MAP-REDUCE"<<std::endl;
  long res2 = map_reduce(p, v.begin(), v.end(),(long) 0,
    [](long x){return 2*x;},
    [](long b, long a){ return b+a;} 
  );
  std::cout<<res2<<std::endl;

  std::cout<<"STENCIL"<<std::endl;
  auto beg = v.begin();
  auto end = v.end();
  stencil(p, v.begin(), v.end(), v2.begin(),
    [](auto it, auto n){ 
      return (*it + std::accumulate(std::begin(n), std::end(n),0))/double(n.size()+1);
    },
    [&beg,&end](auto it) {
      std::vector <double> r;
      if (it != beg) r.push_back(*prev(it));
      if (std::distance(it, end)>1) r.push_back(*std::next(it));
      return r;
    }
  );
  for(int i = 0; i<size;i++) std::cout<<v2[i] << " ";
  std::cout<<std::endl; 
  std::cout<<"D&C"<<std::endl;
  for (int i = 0; i< 20 ; i++){
  auto fib = grppi::divide_conquer(p,
    i,
    [](int x) -> std::vector<int> {
      return { x-1, x-2 };
    },
    [](int x) { return x<2; },
    [](int) { return 1; },
    [](int s1, int s2) {
      return s1+s2;
    }
  );
    std::cout<<i<<": "<<fib<<std::endl; 
  }
  std::cout<<"PIPELINE"<<std::endl;
  int val = 0;
  grppi::pipeline(p,
    [&val]()-> std::experimental::optional<int> {
      if(val < 10 ) return {val++};
      else return {};
    },
    grppi::farm(4,
      grppi::pipeline(
        [](int val) { return val * 2;},
        [](int val) { return val * 2;}
      )
    ),
    [](int val)
    {
      std::cout<< val<<std::endl;
    }
  );

  std::cout<<"FINISHED"<<std::endl; 

}

