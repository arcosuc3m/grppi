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


int main(int argc, char *argv[]){

  std::cout<<"Bin2Stirng: This sample transforms a file or a set of files that contains integer's binary value into files containing their decimal values as strings separated by \\n." <<std::endl;
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
  aspide::binary_container container(argv[3]);

  auto reader = binary_reader<int>(container,[](char * a){
                  int * value = reinterpret_cast<int *>(a);
		  std::cout<<"Value : "<<*value<<std::endl;
                  return *value;
                  }, 4);


  std::cout<< "Output path: "<<argv[4]<<std::endl;
  aspide::output_container out(argv[4]);

  format_writer formatter(out,
    [](int item ) -> std::string{
       std::ostringstream s;
       s << item <<std::endl;
       return s.str();
    }
  );

  grppi::map(exec, reader, formatter, [](int i){ return i;});

  return 0;
}

