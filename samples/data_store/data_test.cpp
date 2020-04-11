#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>

#include <unistd.h>

#include "task_dist/zmq_port_service.h"
#include "task_dist/zmq_data_service.h"


int test_local (grppi::zmq_data_service & data_service, int numTokens)
{
  // test local storage
  std::vector<grppi::zmq_data_reference> ref;
  //for (int i=0; i<numTokens+1; i++) {
  for (int i=0; i<numTokens; i++) {
    try {
       if (i % 2 == 0) {
        ref.emplace_back(data_service.set(i));
        std::cout << "test_local: store " << std::to_string(i);
      } else {
        ref.emplace_back(data_service.set("val" + std::to_string(i)));
        std::cout << "test_local: store " << "val" + std::to_string(i);
      }
      std::cout << " (" << ref[i].get_id() << "," << ref[i].get_pos() << ")" << std::endl;
    } catch (std::runtime_error &e) {
      std::cout << "test_local: RunTime error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "test_local: ERROR: UNO" << e.what() << std::endl;
    }
  }

  for (int i=ref.size()-1; i>=0; i--) {
    try {
      std::cout << "test_local: get: (" << ref[i].get_id() << "," << ref[i].get_pos() << ")" << std::endl;
      if (i % 2 == 0) {
        int aux = data_service.get<int>(ref[i]);
        std::cout << "test_local: 1: ref<int> = " << aux << std::endl;
        //int aux2 = data_service.get<int>(ref[i]);
        //std::cout << "test_local: 2: ref<int> = " << aux2 << std::endl;
      } else {
        std::string aux = data_service.get<std::string>(ref[i]);
        std::cout << "test_local: ref<string> = " << aux << std::endl;
      }
    } catch (std::runtime_error &e) {
      std::cout << "test_local: RunTime error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "test_local: ERROR: DOS" << e.what() << std::endl;
    }
  }
  return 0;
}

int test_remote (grppi::zmq_data_service & data_service, int numTokens, int id)
{
  if (id == 0) {
    // test local storage
    std::vector<grppi::zmq_data_reference> ref;
    //for (int i=0; i<numTokens+1; i++) {
    for (int i=0; i<numTokens; i++) {
      try {
        ref.emplace_back(data_service.set("val" + std::to_string(i)));
        std::cout << "test_remote: store " << "val" + std::to_string(i);
        std::cout << " (" << ref[i].get_id() << "," << ref[i].get_pos() << ")" << std::endl;
      } catch (std::runtime_error &e) {
        std::cout << "test_remote: RunTime error: " << e.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "test_remote: ERROR: TRES" << e.what() << std::endl;
      } catch (...) {
        std::cout << "test_remote: ERROR: TRES" << std::endl;
      }
    }
  }
  
  if (id == 1) {
    //for (int i=0; i<numTokens+1; i++) {
    for (int i=0; i<numTokens; i++) {
      try {
        grppi::zmq_data_reference ref(0,i);
        std::cout << "get: (" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
        std::string aux = data_service.get<std::string>(ref);
        std::cout << "ref<string> = " << aux << std::endl;
      } catch (std::runtime_error &e) {
        std::cout << "RunTime error: " << e.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "ERROR: CUATRO" << e.what() << std::endl;
      } catch (...) {
        std::cout << "ERROR: CUATRO" << std::endl;
      }
    }
  }
  return 0;
}

int main (int argc, char *argv[])
{
    std::map <int,std::string> machines;
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " id machine [...]" << std::endl;
        return -1;
    }
    int id = atoi(argv[1]);
    for (int i=2; i<argc; i++) {
        machines[i-2]=argv[i];
    }
    int numTokens = 10;
    
    std::cout << "main: Number of machines: " << machines.size() << std::endl;
    for (int i=0; i<(int)machines.size(); i++) {
        std::cout << "main: Machines " << i << ": " << machines[i] << std::endl;
    }
    std::cout << "main: This machine: " << id << " -> " << machines[id] << std::endl;
    std::cout << "main: Number of tokens: " << numTokens << std::endl;
    
    std::cout << "main: init port service: " << numTokens << std::endl;
    auto port_service = std::make_shared<grppi::zmq_port_service>(machines[0], 5570, (id==0));

    std::cout << "main: init data service: " << numTokens << std::endl;
    //grppi::zmq_data_service data_service{};
    grppi::zmq_data_service data_service(machines, id, port_service, numTokens*2);


    std::cout << "main: local test: " << numTokens << std::endl;
    test_local (data_service, numTokens);

    std::cout << "main: remote test: " << numTokens << std::endl;
    test_remote (data_service, numTokens, id);

    sleep(10);
    return 0;
}
