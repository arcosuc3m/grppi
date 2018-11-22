#include <iostream>
#include <vector>

#include "grppi.h"
#include "task/parallel_execution_task.h"
#include "task/fifo_scheduler.h"
#include "task/simple_task.h"
using namespace grppi;
int main(){
  fifo_scheduler<simple_task> sched;
  parallel_execution_task<fifo_scheduler<simple_task>> p{sched};
  std::vector<int> v(24);
  for(int i = 0; i<24;i++) v[i] = 1;
//  int res = reduce(p, v.begin(), v.end(), 0, [](int b, int a){ return b+a;});
  int res = map_reduce(p, v.begin(), v.end(), 0, [](int x){return 2*x;},[](int b, int a){ return b+a;});
 
//  for(int i = 0; i<48;i++) std::cout<<v[i] << " ";
  
  std::cout<<res<<std::endl; 
}

