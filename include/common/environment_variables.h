#ifndef COMMON_ENVIRONMENT_VARIABLES
#define COMMON_ENVIRONMENT_VARIABLES

#include <sstream>
#include <thread>
#include <cstdlib>
#include <cstring>

namespace grppi
{

  int default_concurrency_degree =
      [](){
        if(const char* env_value = std::getenv("GRPPI_NUM_THREADS"))
        {
           std::istringstream env_string(env_value);
           int value;
           env_string >> value;
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
           std::istringstream env_string(env_value);
           int value;
           env_string >> value;
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

}

#endif
