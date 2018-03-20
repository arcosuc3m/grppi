#ifndef COMMON_ENVIRONMENT_VARIABLES
#define COMMON_ENVIRONMENT_VARIABLES

#include "mpmc_queue.h"

#include <sstream>
#include <thread>
#include <cstring>
#include <string>

namespace grppi
{

class environment_variables{
 public:
  int default_concurrency_degree =
      [](){
        if(const char* env_value = std::getenv("GRPPI_NUM_THREADS"))
        {
           int value = std::stoi(env_value);
           return value;
        }
         return static_cast<int>(std::thread::hardware_concurrency());
      }();

  bool default_ordering =
     [](){
       if(const char* env_value = std::getenv("GRPPI_ORDERING"))
       {
          if(strcmp(env_value,"TRUE") != 0) return false;
       }
       return true;
     }();

  int default_queue_size =
    [](){
      if(const char* env_value = std::getenv("GRPPI_QUEUE_SIZE"))
        {
           int value = std::stoi(env_value);
           return value;
        }
        return 100;
    }();

  queue_mode default_queue_mode =
     [](){
      if(const char* env_value = std::getenv("GRPPI_QUEUE_MODE"))
       {
          if(strcmp(env_value,"lockfree") != 0) return queue_mode::lockfree;
       }
       return queue_mode::blocking;
     }();

  std::string default_dynamic_execution =
    []() -> std::string{
      if(const char* env_value = std::getenv("GRPPI_DYN_EXECUTION"))
       {
         return {env_value};
       }
       return {"seq"};
    }();

};
}

#endif
