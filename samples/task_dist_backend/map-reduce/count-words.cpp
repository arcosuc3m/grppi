#undef NDEBUG

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
#include <boost/serialization/map.hpp>
#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

using namespace grppi;

  std::cout<<"Count-Words: This sample computes the number of occurrences of each word part of a file or a set of files.
  if (argc != 5) {
    std::cout << "Usage: " << argv[0] << " <node id> <server id> <input_dir> <output_dir>" << std::endl;
  }
  long id = atoi(argv[1]);
  long server_id = atoi(argv[2]);
  bool is_server = (id == server_id);

  std::map<long, std::string> machines{{0, "127.0.0.1"}};
  //std::map<long, std::string> machines{{0, "127.0.0.1"},{1, "127.0.0.1"}};
  //std::map<long, std::string> machines{{0, "127.0.0.1"},{1, "192.168.1.37"}};
  //std::map<long, std::string> machines{{0, "127.0.0.1"},{1, "172.16.83.183"}};
  auto port_serv = std::make_shared<zmq_port_service> (machines[0], 5570, is_server);
  std::cout << "port_service_->new_port() : " << port_serv->new_port() << std::endl;
  auto sched = std::make_unique<zmq_scheduler<zmq_task>>(machines, id,
                                                        port_serv, 100, server_id, 2);
  parallel_execution_dist_task<zmq_scheduler<zmq_task>> exec{std::move(sched)};

  std::cout<< "Input file or collection: "<<argv[3]<<std::endl;
  aspide::text_in_container container(argv[3],'\n');

  std::cout<< "Output path: "<<argv[4]<<std::endl;
  aspide::output_container out(argv[4]);


  format_writer formatter(out,
    [](std::map<std::string,int> item ) -> std::string{
       std::ostringstream s;
       for (auto & w : item) {
          s << w.first<<" "<<w.second<<std::endl;
       }
       return s.str();
    }
  );

  grppi::map_reduce(exec,container, formatter,
                  [](std::string s)-> std::map<std::string,int>
                  {
                      return {{s,1}};
                  },
                  [](std::map<std::string,int> &lhs,
                      std::map<std::string,int> &rhs)
                  {
                     for (auto & w : rhs) {
                       lhs[w.first]+= w.second;
                     }
                     return lhs;
                  });

