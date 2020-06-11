#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>

#include <unistd.h>

#include "task_dist/zmq_port_service.h"
#include "task_dist/zmq_data_service.h"
#include <boost/serialization/vector.hpp>


long test_local (grppi::zmq_data_service & data_service, long numTokens)
{
  std::cout << "test_local:  BEGIN\n";
  // test local storage
  std::vector<grppi::zmq_data_reference> ref;
  for (long i=0; i<numTokens+1; i++) {
    try {
       if (i % 2 == 0) {
        ref.emplace_back(data_service.set(std::move(i)));
        std::cout << "test_local: store " << std::to_string(i);
      } else {
        std::vector<unsigned char> aux_vec{'v','a','l',('0'+(unsigned char)i)};
        void * ptr = (void *)(aux_vec.data());
        ref.emplace_back(data_service.set(std::move(aux_vec)));
        std::cout << "test_local: store " << "val" + std::to_string(i) << ", ptr:" << ptr;
      }
      std::cout << " (" << ref[i].get_id() << "," << ref[i].get_pos() << ")" << std::endl;
    } catch (std::runtime_error &e) {
      std::cout << "test_local: RunTime error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "test_local: ERROR: UNO" << e.what() << std::endl;
    }
  }

  for (long i=ref.size()-1; i>=0; i--) {
    try {
      std::cout << "test_local: get: (" << ref[i].get_id() << "," << ref[i].get_pos() << ")" << std::endl;
      if (i % 2 == 0) {
        long aux = data_service.get<long>(ref[i],true);
        std::cout << "test_local: 1: ref<long> = " << aux << std::endl;
      } else {
        std::vector<unsigned char> aux = data_service.get<std::vector<unsigned char>>(ref[i],true);
        void * ptr = (void *)(aux.data());
        std::cout << "test_local: ref<string> = " << (char *)(aux.data()) << ", ptr:" << ptr << std::endl;
      }
    } catch (std::runtime_error &e) {
      std::cout << "test_local: RunTime error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "test_local: ERROR: DOS" << e.what() << std::endl;
    }
  }
  
  try {
    std::cout << "test_local: initial set" << std::endl;
    std::string old_str{"HELLO"};
    grppi::zmq_data_reference old_ref = data_service.set(std::move(old_str));
    std::cout << "test_local: set<string>: " << old_str << std::endl;
    std::string new_str = data_service.get<std::string>(old_ref,false);
    std::cout << "test_local: get<string,false>: " << new_str << std::endl;
    std::cout << "test_local: (old_str == new_str): " << (old_str == new_str) << std::endl;

    std::cout << "test_local: already allocated set" << std::endl;
    long old_num{10};
    grppi::zmq_data_reference new_ref = data_service.set(std::move(old_num), old_ref);
    std::cout << "test_local: set<long>: " << old_num << std::endl;
    std::cout << "test_local: (new_ref == old_ref): " << (new_ref == old_ref) << std::endl;
    long new_num = data_service.get<long>(old_ref,false);
    std::cout << "test_local: get<long,false>: " << new_num << std::endl;
    std::cout << "test_local: (old_num == new_num): " << (old_num == new_num) << std::endl;
    long new2_num = data_service.get<long>(old_ref,true);
    std::cout << "test_local: get<long,true>: " << new2_num << std::endl;
    std::cout << "test_local: (new_num == new2_num): " << (new_num == new2_num) << std::endl;
    std::cout << "test_local: try to get<long,false>: (should fail)"<< std::endl;
    long new3_num = data_service.get<long>(old_ref,false);
  } catch (std::runtime_error &e) {
    std::cout << "test_local: RunTime error: " << e.what() << std::endl;
  } catch (std::exception &e) {
    std::cout << "test_local: ERROR: TRES" << e.what() << std::endl;
  } catch (...) {
    std::cout << "test_local: ERROR: TRES" << std::endl;
  }
  
  try {
    std::string old_str{"HELLO"};
    grppi::zmq_data_reference old_ref = data_service.set(std::move(old_str));
    std::cout << "test_local: set<string>: " << old_str << std::endl;

    std::string new_str = data_service.get<std::string>(old_ref,false,2);
    std::cout << "test_local: get<string,false,2>: " << new_str << std::endl;
    std::cout << "test_local: (old_str == new_str): " << (old_str == new_str) << std::endl;

    std::string new2_str = data_service.get<std::string>(old_ref,true,2);
    std::cout << "test_local: get<string,true,2>: " << new2_str << std::endl;
    std::cout << "test_local: (old_str == new2_str): " <<(old_str == new2_str) << std::endl;

    std::string new3_str = data_service.get<std::string>(old_ref,true,2);
    std::cout << "test_local: get<string,true,2>: " << new3_str << std::endl;
    std::cout << "test_local: (old_str == new3_str): " <<(old_str == new3_str) << std::endl;

    std::cout << "test_local: try to get<string,true,2>: (should fail)"<< std::endl;
    std::string new4_str = data_service.get<std::string>(old_ref,true,2);
  
  } catch (std::runtime_error &e) {
    std::cout << "test_local: RunTime error: " << e.what() << std::endl;
  } catch (std::exception &e) {
    std::cout << "test_local: ERROR: CUATRO" << e.what() << std::endl;
  } catch (...) {
    std::cout << "test_local: ERROR: CUATRO" << std::endl;
  }

 std::cout << "test_local:  END\n";
 return 0;
}

long test_remote (grppi::zmq_data_service & data_service, long numTokens, long id)
{
  std::cout << "test_remote:  BEGIN\n";
  if (id == 0) {
    std::cout << "test_remote:  storing\n";
    // test local storage
    std::vector<grppi::zmq_data_reference> ref;
    for (long i=0; i<numTokens+1; i++) {
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
    std::cout << "test_remote:  getting\n";
    //for (long i=0; i<numTokens+1; i++) {
    for (long i=0; i<numTokens; i++) {
      try {
        grppi::zmq_data_reference ref(0,i);
        std::cout << "get: (" << ref.get_id() << "," << ref.get_pos() << ")" << std::endl;
        std::string aux = data_service.get<std::string>(ref,true);
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
  std::cout << "test_remote:  END\n";
  return 0;
}

long test_remote_2 (grppi::zmq_data_service & data_service, long numTokens, long id)
{
  std::cout << "test_remote_2:  BEGIN\n";
  if (id == 0) {
    try {
      std::cout << "test_remote_2: initial set" << std::endl;
      std::string old_str{"HELLO"};
      grppi::zmq_data_reference old_ref = data_service.set(std::move(old_str));
      std::cout << "test_remote_2: set<string>: " << old_str << std::endl;
      std::string new_str = data_service.get<std::string>(old_ref,false);
      std::cout << "test_remote_2: get<string,false>: " << new_str << std::endl;
      std::cout << "test_remote_2: (old_str == new_str): " << (old_str == new_str) << std::endl;
    } catch (std::runtime_error &e) {
      std::cout << "test_remote_2: RunTime error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "test_remote_2: ERROR: UNO" << e.what() << std::endl;
    } catch (...) {
      std::cout << "test_remote_2: ERROR: UNO" << std::endl;
    }
    
    try {
      std::string old_str{"HELLO"};
      grppi::zmq_data_reference old_ref = data_service.set(std::move(old_str));
      std::cout << "test_remote_2: set<string>: " << old_str << std::endl;

      std::string new_str = data_service.get<std::string>(old_ref,false,2);
      std::cout << "test_remote_2: get<string,false,2>: " << new_str << std::endl;
      std::cout << "test_remote_2: (old_str == new_str): " << (old_str == new_str) << std::endl;
    } catch (std::runtime_error &e) {
      std::cout << "test_remote_2: RunTime error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "test_remote_2: ERROR: DOS" << e.what() << std::endl;
    } catch (...) {
      std::cout << "test_remote_2: ERROR: DOS" << std::endl;
    }
  } else if (id == 1) {
    try {
      grppi::zmq_data_reference old_ref(0,0);
      std::string new_str = data_service.get<std::string>(old_ref,false);
      std::cout << "test_remote_2: get<string,false>: " << new_str << std::endl;
      std::cout << "test_remote_2: already allocated set" << std::endl;
      long old_num{10};
      grppi::zmq_data_reference new_ref = data_service.set(std::move(old_num), old_ref);
      std::cout << "test_remote_2: set<long>: " << old_num << std::endl;
      std::cout << "test_remote_2: (new_ref == old_ref): " << (new_ref == old_ref) << std::endl;
      long new_num = data_service.get<long>(old_ref,false);
      std::cout << "test_remote_2: get<long,false>: " << new_num << std::endl;
      std::cout << "test_remote_2: (old_num == new_num): " << (old_num == new_num) << std::endl;
      long new2_num = data_service.get<long>(old_ref,true);
      std::cout << "test_remote_2: get<long,true>: " << new2_num << std::endl;
      std::cout << "test_remote_2: (new_num == new2_num): " << (new_num == new2_num) << std::endl;
      std::cout << "test_remote_2: try to get<long,false>: (should fail)"<< std::endl;
      long new3_num = data_service.get<long>(old_ref,false);
    } catch (std::runtime_error &e) {
      std::cout << "test_remote_2: RunTime error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "test_remote_2: ERROR: UNO" << e.what() << std::endl;
    } catch (...) {
      std::cout << "test_remote_2: ERROR: UNO" << std::endl;
    }
    
    try {
      grppi::zmq_data_reference old_ref(0,1);
      std::string new2_str = data_service.get<std::string>(old_ref,true,2);
      std::cout << "test_remote_2: get<string,true,2>: " << new2_str << std::endl;

      std::string new3_str = data_service.get<std::string>(old_ref,true,2);
      std::cout << "test_remote_2: get<string,true,2>: " << new3_str << std::endl;
      std::cout << "test_remote_2: (new2_str == new3_str): " <<(new2_str == new3_str) << std::endl;

      std::cout << "test_remote_2: try to get<string,true,2>: (should fail)"<< std::endl;
      std::string new4_str = data_service.get<std::string>(old_ref,true,2);
  
    } catch (std::runtime_error &e) {
      std::cout << "test_remote_2: RunTime error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "test_remote_2: ERROR: DOS" << e.what() << std::endl;
    } catch (...) {
      std::cout << "test_remote_2: ERROR: DOS" << std::endl;
    }
  }
  std::cout << "test_remote_2:  END\n";
  return 0;
}

int main (int argc, char *argv[])
{
    std::map <long,std::string> machines;
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " id machine [...]" << std::endl;
        return -1;
    }
    long id = atoi(argv[1]);
    for (long i=2; i<argc; i++) {
        machines[i-2]=argv[i];
    }
    long numTokens = 10;
    
    std::cout << "main: Number of machines: " << machines.size() << std::endl;
    for (long i=0; i<(long)machines.size(); i++) {
        std::cout << "main: Machines " << i << ": " << machines[i] << std::endl;
    }
    std::cout << "main: This machine: " << id << " -> " << machines[id] << std::endl;
    std::cout << "main: Number of tokens: " << numTokens << std::endl;
    
    std::cout << "main: init port service: " << numTokens << std::endl;
    auto port_service = std::make_shared<grppi::zmq_port_service>(machines[0], 5570, (id==0));

    std::cout << "main: init data service: " << numTokens << std::endl;

    try {
        grppi::zmq_data_service data_service(machines, id, port_service, numTokens);

        std::cout << "main: local test: " << numTokens << std::endl;
        test_local (data_service, numTokens);
      } catch (...) {
        std::cout << "ERROR: MAIN LOCAL" << std::endl;
      }

    try {
        grppi::zmq_data_service data_service(machines, id, port_service, numTokens);

        std::cout << "main: remote test 1: " << numTokens << std::endl;
        test_remote (data_service, numTokens, id);
        sleep(5);
      } catch (...) {
        std::cout << "ERROR: MAIN REMOTE 1" << std::endl;
      }

    try {
        grppi::zmq_data_service data_service(machines, id, port_service, numTokens);

        std::cout << "main: remote test 2: " << numTokens << std::endl;
        test_remote_2 (data_service, numTokens, id);
        sleep(5);
      } catch (...) {
        std::cout << "ERROR: MAIN REMOTE 1" << std::endl;
      }

    return 0;
}
