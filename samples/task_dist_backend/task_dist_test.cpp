#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
//#include <experimental/optional>
#include <optional>
#include <stdlib.h>     /* atoi */


#include <string.h>
//#include "text_in_container.hpp"
//#include "file_generic.hpp"

#include "grppi.h"
#include "task_dist/zmq_port_service.h"
#include "task_dist/zmq_task.h"


using namespace grppi;

// data for scheduler debugging
std::vector<zmq_task> conf_tasks {
        {0,0,0,std::vector<long>{0}, true}, // generator (seq 0)
        {1,0,0,std::vector<long>{0,1}, true}, // pipeline-middle-1   (seq 1)
        {2,0,0,std::vector<long>{0,1}, true}, // farm (par)
        {3,0,0,std::vector<long>{0}, true}, // filter  (NODEBUG) (seq 0)
        {4,0,0,std::vector<long>{0}, false}, // farm-pipeline-init (par) (UNCH)
        {5,0,0,std::vector<long>{0}, true}, // farm-pipeline-middle (par)
        {6,0,0,std::vector<long>{1}, true}, // farm-pipeline-end (par)
        {7,0,0,std::vector<long>{0}, true}, // steam-reduce (seq 1) (NODEBUG+1)
        {8,0,0,std::vector<long>{0}, false}, // iterator  (seq 0)(NODEBUG+1)(UNCH)
        {9,0,0,std::vector<long>{0}, false}}; // consumer (seq 1) (UNCH)

std::vector<zmq_task> global_tasks;


int main(int argc, char *argv[]){

  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <node id> <server id>" << std::endl;
  }
  long id = atoi(argv[1]);
  long server_id = atoi(argv[2]);
  bool is_server = (id == server_id);
 
  std::cout << "node_id = " << id << ", server_id = " << server_id << std::endl;

  std::map<long, std::string> machines{{0, "127.0.0.1"}};
  //std::map<long, std::string> machines{{0, "127.0.0.1"},{1, "127.0.0.1"}};
  //std::map<long, std::string> machines{{0, "127.0.0.1"},{1, "192.168.1.37"}};
  //std::map<long, std::string> machines{{0, "127.0.0.1"},{1, "172.16.83.183"}};
  auto port_serv = std::make_shared<zmq_port_service> (machines[0], 5570, is_server);
  std::cout << "port_service_->new_port() : " << port_serv->new_port() << std::endl;
  auto sched = std::make_unique<zmq_scheduler<zmq_task>>(machines, id,
                                                        port_serv, 100, server_id, 2);
  parallel_execution_dist_task<zmq_scheduler<zmq_task>> exec{std::move(sched)};

  aspide::text_in_container container("file:/./datos",'\n');
  //return 0;
  std::cout<<"---PIPELINE---"<<std::endl;
  long val = 0;
  grppi::pipeline(exec,
/*    [&val]()-> std::experimental::optional<long> {
    [&val](std::vector<zmq_task> &task = global_tasks)-> std::experimental::optional<long> {
      task.push_back(conf_tasks[0]);
      task.push_back(conf_tasks[1]);
      std::ostringstream ss;
      ss << "--------------- stage 0: generator val++ = " << ((val < 2) ? (val) : -1) << "---------------" << std::endl;
      std::cout << ss.str();
      if(val < 2 ) return {val++};
      else return {};
    },*/
    container,
    grppi::farm(4,[](std::string s,std::vector<zmq_task> &task = global_tasks){
        task.push_back(conf_tasks[0]);
        task.push_back(conf_tasks[1]);
             return stoi(s);
	    }),
    [](long val, std::vector<zmq_task> &task = global_tasks){
        task.push_back(conf_tasks[1]);
        task.push_back(conf_tasks[2]);
        std::ostringstream ss;
        ss << "--------------- stage 1: pipeline_middle_1 val*2 = (" << val << ", " << (val*2  == 0) <<  ") ---------------" << std::endl;
        std::cout << ss.str();
        return val*2 == 0;
    },
    grppi::farm(4,
      [](long val, std::vector<zmq_task> &task = global_tasks){
        task.push_back(conf_tasks[2]);
        task.push_back(conf_tasks[3]);
        std::ostringstream ss;
        ss << "--------------- stage 2: pipeline_middle_2 val/2 = (" << val << ", " << (val/2  == 0) <<  ") ---------------" << std::endl;
        std::cout << ss.str();
        return val/2 == 0;
    }),
    grppi::discard([](long val, std::vector<zmq_task> &task = global_tasks){
        task.push_back(conf_tasks[3]);
        task.push_back(conf_tasks[4]);
        std::ostringstream ss;
        ss << "--------------- stage 3: discard val%2 = (" << val << ", " << (val%2  == 0) <<  ") ---------------" << std::endl;
        std::cout << ss.str();
        return val%2 == 0;
    }),
    grppi::farm(4,
      grppi::pipeline(
        [](long val, std::vector<zmq_task> &task = global_tasks) {
            task.push_back(conf_tasks[4]);
            task.push_back(conf_tasks[5]);
            std::ostringstream ss;
            ss << "--------------- stage 4: farm_pipeline_init val * 2 = (" << val * 2 << ") ---------------" << std::endl;
            std::cout << ss.str();
            return val * 2;
        },
        [](long val, std::vector<zmq_task> &task = global_tasks) {
            task.push_back(conf_tasks[5]);
            task.push_back(conf_tasks[6]);
            std::ostringstream ss;
            ss << "--------------- stage 5: farm_pipeline_middle val * 4 = (" << val * 4 << ") ---------------" << std::endl;
            std::cout << ss.str();
            return val * 4;
        },
        [](long val, std::vector<zmq_task> &task = global_tasks) {
            task.push_back(conf_tasks[6]);
            task.push_back(conf_tasks[7]);
            std::ostringstream ss;
            ss << "--------------- stage 6: farm_pipeline_end val / 2 = (" << val / 2 << ") ---------------" << std::endl;
            std::cout << ss.str();
            return val / 2;
        }
      )
    ),
    grppi::reduce(2,1,0,[](long a, long b, std::vector<zmq_task> &task = global_tasks) {
        task.push_back(conf_tasks[7]);
        task.push_back(conf_tasks[8]);
        std::ostringstream ss;
        ss << "--------------- stage 7: reduce a + b  = (" << a << " + " << b << " = " << a + b << ") ---------------" << std::endl;
        std::cout << ss.str();
        return a+b;
    }),
    grppi::repeat_until([](long val, std::vector<zmq_task> &task = global_tasks) {
        task.push_back(conf_tasks[8]);
        task.push_back(conf_tasks[9]);
        std::ostringstream ss;
        ss << "--------------- stage 8.1: repeat val * 2 = (" << val * 2 << ") ---------------" << std::endl;
        std::cout << ss.str();
        return val * 2; } , [](long val){
        std::ostringstream ss;
        ss << "--------------- stage 8.2: until val > 50 = (" << val << ", " << (val > 50) << ") ---------------" << std::endl;
        std::cout << ss.str();
        return val > 50;
    }),
    [](long val) {
      std::ostringstream ss;
      ss << "--------------- stage 9: consume val = (" << val << ") ---------------" << std::endl;
      std::cout << ss.str();
    }
  );

  std::cout<<"FINISHED"<<std::endl; 

}

