#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <thread>         // std::thread
#include <mutex>          // std::mutex

#include <unistd.h>
#include "task_dist/zmq_port_service.h"


void thread_func(std::string machine) {
  grppi::zmq_port_service port_service(machine, 5570, false);
  long port=0;
  for (long i=0; i<2; i++) {
    for (long j=4; j<8; j++) {
      try {
          port_service.set(i,j,port);
          std::cout << "set: machine_id=" << i << " key=" << j << " port=" << port << std::endl;
      } catch (std::runtime_error &e) {
        std::cout << "RunTime error: " << e.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "ERROR: UNO" << e.what() << std::endl;
      }
      port++;
      sleep (1);
    }
  }
  return;
}
long test_recv_local(grppi::zmq_port_service &port_service)
{
  long port;
  for (long i=0; i<2; i++) {
    for (long j=4; j<8; j++) {
      try {
          port = port_service.get(i,j,true);
          std::cout << "get: machine_id=" << i << " key=" << j << " port=" << port << std::endl;
      } catch (std::runtime_error &e) {
        std::cout << "RunTime error: " << e.what();
        std::cout << " machine_id=" << i << " key=" << j << " port=" << port << std::endl;

      } catch (std::exception &e) {
        std::cout << "ERROR: UNO" << e.what();
        std::cout << " machine_id=" << i << " key=" << j << " port=" << port << std::endl;
      }
    }
  }
  return 0;
}


long test_local(grppi::zmq_port_service &port_service) {

  // test local storage
  long port=0;
  for (long i=0; i<2; i++) {
    for (long j=0; j<4; j++) {
      try {
          port_service.set(i,j,port);
          std::cout << "set: machine_id=" << i << " key=" << j << " port=" << port << std::endl;
      } catch (std::runtime_error &e) {
        std::cout << "RunTime error: " << e.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "ERROR: UNO" << e.what() << std::endl;
      }
      port++;
    }
  }

  std::cout << std::endl << "------------------" << std::endl << std::endl;

  for (long i=0; i<3; i++) {
    for (long j=0; j<5; j++) {
      try {
          port = port_service.get(i,j,false);
          std::cout << "get: machine_id=" << i << " key=" << j << " port=" << port << std::endl;
      } catch (std::runtime_error &e) {
        std::cout << "RunTime error: " << e.what();
        std::cout << " machine_id=" << i << " key=" << j << " port=" << port << std::endl;

      } catch (std::exception &e) {
        std::cout << "ERROR: UNO" << e.what();
        std::cout << " machine_id=" << i << " key=" << j << " port=" << port << std::endl;
      }
    }
  }
  return 0;
}


int main (int argc, char *argv[])
{
  std::vector<std::string> machines;
  if (argc < 3) {
      std::cout << "Usage: " << argv[0] << " id machine [...]" << std::endl;
      return -1;
  }
  long id = atoi(argv[1]);
  for (long i=2; i<argc; i++) {
      machines.emplace_back(argv[i]);
  }
  long numTokens = 10;
    
  std::cout << "Number of machines: " << machines.size() << std::endl;
  for (long i=0; i<(long)machines.size(); i++) {
      std::cout << "Machines " << i << ": " << machines[i] << std::endl;
  }
  std::cout << "This machine: " << id << " -> " << machines[id] << std::endl;
  std::cout << "Number of tokens: " << numTokens << std::endl;

  grppi::zmq_port_service port_service(machines[0], 5570, true);

  // local test
  test_local(port_service);

  // wait test
  std::thread thread(&thread_func, machines[0]);
  test_recv_local(port_service);
  thread.join();
  return 0;
}
