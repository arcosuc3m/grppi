/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Standard library
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <stdlib.h>
#include <optional>

// grppi
#include "grppi.h"
#include "task_dist/zmq_port_service.h"
#include "task_dist/zmq_task.h"

using namespace grppi;


int main(int argc, char **argv) {
    
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " num <node id> <server id>" << std::endl;
    return 1;
  }
  long num = atoi(argv[1]);
  long id = atoi(argv[2]);
  long server_id = atoi(argv[3]);
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


  // divide and conquer parallel patern
  std::cout<<"---DIVIDE & CONQUER---"<<std::endl;
  auto res = grppi::divide_conquer(exec,
    num,
    [](int x) -> vector<int> {
      std::ostringstream ss;
      ss << "--------------- DIVIDE: " << x << " = (" << x-1 << ", " << x-2 << ") ---------------" << std::endl;
      std::cout << ss.str();
      return { x-1, x-2 };
    },
    [](int x) {
      std::ostringstream ss;
      ss << "--------------- PREDICATE: (" << x << " < 2) = " << (x<2) << " ---------------" << std::endl;
      std::cout << ss.str();
      return x<2;
    },
    [](int x) {
      std::ostringstream ss;
      ss << "--------------- SOLVE: " << x << " -> 1 ---------------" << std::endl;
      std::cout << ss.str();
      return 1;
    },
    [](int s1, int s2) {
      std::ostringstream ss;
      ss << "--------------- COMBINE: " << s1 << " + " << s2 << " = " << s1 + s2 << " ---------------" << std::endl;
      std::cout << ss.str();
      return s1+s2;
    }
  );
  std::cout << "Fibonacci(" << num << ")= " << res << std::endl;
  std::cout<<"FINISHED"<<std::endl;
return 0;
}
