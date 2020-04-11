#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
#include <experimental/optional>
#include <stdlib.h>     /* atoi */

#include "grppi.h"
#include "task_dist/zmq_port_service.h"
#include "task_dist/zmq_task.h"

using namespace grppi;

int main(int argc, char *argv[]){

  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <node id> <server id>" << std::endl;
  }
  int id = atoi(argv[1]);
  int server_id = atoi(argv[2]);
  bool is_server = (id == server_id);
 
  std::cout << "node_id = " << id << ", server_id = " << server_id << std::endl;

  //std::map<int, std::string> machines{{0, "127.0.0.1"}};
  //std::map<int, std::string> machines{{0, "127.0.0.1"},{1, "127.0.0.1"}};
  std::map<int, std::string> machines{{0, "127.0.0.1"},{1, "172.16.83.183"}};
  auto port_serv = std::make_shared<zmq_port_service> (machines[0], 5570, is_server);
  std::cout << "port_service_->new_port() : " << port_serv->new_port() << std::endl;
  auto sched = std::make_unique<zmq_scheduler<zmq_task>>(machines, id,
                                                        port_serv, 100, server_id);
  parallel_execution_dist_task<zmq_scheduler<zmq_task>> p{std::move(sched)};

  //return 0;
  std::cout<<"---PIPELINE---"<<std::endl;
  int val = 0;
  grppi::pipeline(p,
    [&val]()-> std::experimental::optional<int> {
      std::ostringstream ss;
      ss << "--------------- stage 1: generator val++ = " << ((val < 10) ? (val) : -1) << "---------------" << std::endl;
      std::cout << ss.str();
      if(val < 10 ) return {val++};
      else return {};
    },
    grppi::discard([](int val){
        std::ostringstream ss;
        ss << "--------------- stage 2: discard val%2 = (" << val << ", " << (val%2  == 0) <<  ") ---------------" << std::endl;
        std::cout << ss.str();
        return val%2 == 0; }),
    grppi::farm(4,
      grppi::pipeline(
        [](int val) {
            std::ostringstream ss;
            ss << "--------------- stage 3: val * 2 = (" << val * 2 << ") ---------------" << std::endl;
            std::cout << ss.str();
            return val * 2;},
        [](int val) {
            std::ostringstream ss;
            ss << "--------------- stage 4: val * 2 = (" << val * 2 << ") ---------------" << std::endl;
            std::cout << ss.str();
            return val * 2;}
      )
    ),
    grppi::reduce(2,1,0,[](int a, int b) {
        std::ostringstream ss;
        ss << "--------------- stage 5: reduce a + b  = (" << a << " + " << b << " = " << a + b << ") ---------------" << std::endl;
        std::cout << ss.str();
        return a+b; } ),
    grppi::repeat_until([](int val) {
        std::ostringstream ss;
        ss << "--------------- stage 6.1: repeat val * 2 = (" << val * 2 << ") ---------------" << std::endl;
        std::cout << ss.str();
        return val * 2; } , [](int val){
        std::ostringstream ss;
        ss << "--------------- stage 6.2: until val > 50 = (" << val << ", " << (val > 50) << ") ---------------" << std::endl;
        std::cout << ss.str();
        return val > 50;}),
    [](int val)
    {
      std::ostringstream ss;
      ss << "--------------- stage 7: consume val = (" << val << ") ---------------" << std::endl;
      std::cout << ss.str();
    }
  );

  std::cout<<"FINISHED"<<std::endl; 

}

