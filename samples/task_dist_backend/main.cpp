#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
#include <experimental/optional>

#include "grppi.h"

using namespace grppi;

int main(){

  auto sched = std::make_shared<zmq_scheduler<zmq_task>>();
  parallel_execution_dist_task<zmq_scheduler<zmq_task>> p{sched};

  std::cout<<"---PIPELINE---"<<std::endl;
  int val = 0;
  grppi::pipeline(p,
    [&val]()-> std::experimental::optional<int> {
      std::cout << "stage 1" << std::endl;
      if(val < 10 ) return {val++};
      else return {};
    },
    grppi::discard([](int val){
        std::cout << "stage 2" << std::endl;
        return val%2 == 0; }),
    grppi::farm(4,
      grppi::pipeline(
        [](int val) {
            std::cout << "stage 3" << std::endl;
            return val * 2;},
        [](int val) {
            std::cout << "stage 4" << std::endl;
            return val * 2;}
      )
    ),
    grppi::reduce(2,1,0,[](int a, int b) {
        std::cout << "stage 5" << std::endl;
        return a+b; } ),
    grppi::repeat_until([](int val) {
        std::cout << "stage 6" << std::endl;
        return val * 2; } , [](int val){
        std::cout << "stage 7" << std::endl;
        return val > 50;}),
    [](int val)
    {
      std::cout << "stage 8" << std::endl;
      std::cout<< val<<std::endl;
    }
  );

  std::cout<<"FINISHED"<<std::endl; 

}

