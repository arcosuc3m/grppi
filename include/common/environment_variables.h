#ifndef COMMON_ENVIRONMENT_VARIABLES
#define COMMON_ENVIRONMENT_VARIABLES

#include "mpmc_queue.h"

#include <iostream>
#include <sstream>
#include <thread>
#include <cstring>
#include <string>

namespace grppi
{

class environment_variables{
 private:
   void get_concurrency_degree (){
     if(const char* env_value = std::getenv("GRPPI_NUM_THREADS"))
     {
       try{
         int value = std::stoi(env_value);
         default_concurrency_degree = value;
       } catch(int e){
         std::cerr << "Non valid argument for number of threads" << std::endl;
       }
     }    
   }

   void get_ordering(){
     if(const char* env_value = std::getenv("GRPPI_ORDERING"))
     {
       if(strcmp(env_value,"TRUE") != 0) 
         default_ordering = false;
     }
   }

   void get_queue_size(){
     if(const char* env_value = std::getenv("GRPPI_QUEUE_SIZE"))
     {
       try{
         int value = std::stoi(env_value);
         default_queue_size = value;
       } catch (int e){
         std::cerr << "Non valid argument for queue size" << std::endl;
       }
     }
   }

   void get_queue_mode(){
     if(const char* env_value = std::getenv("GRPPI_QUEUE_MODE"))
     {
       if(strcmp(env_value,"lockfree") != 0) 
         default_queue_mode = queue_mode::lockfree;   
     }
   }
  
   void get_dynamic_execution(){
     if(const char* env_value = std::getenv("GRPPI_DYN_EXECUTION"))
     {
       default_dynamic_execution = env_value;
     }
   }

 public:
   environment_variables() :
    default_concurrency_degree(static_cast<int>(std::thread::hardware_concurrency())),
    default_ordering(true),
    default_queue_size(100),
    default_queue_mode(queue_mode::blocking),
    default_dynamic_execution("seq")
  { 
    get_concurrency_degree();
    get_ordering();
    get_queue_size();
    get_queue_mode();
    get_dynamic_execution();
  }

  int default_concurrency_degree;
  bool default_ordering;
  int default_queue_size;
  queue_mode default_queue_mode;
  std::string default_dynamic_execution;

};
}

#endif
