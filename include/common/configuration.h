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
#ifndef GRPPI_COMMON_CONFIGURATION
#define GRPPI_COMMON_CONFIGURATION

#include "mpmc_queue.h"

#include <iostream>
#include <sstream>
#include <thread>
#include <cstring>
#include <string>

namespace grppi {

class configuration {
public:

  int concurrency_degree() const noexcept {
    return concurrency_degree_;
  }

  bool ordering() const noexcept {
    return ordering_;
  }

  int queue_size() const noexcept {
    return queue_size_;
  }

  queue_mode mode() const noexcept {
    return queue_mode_;
  }

  std::string dynamic_execution() const noexcept {
    return dynamic_execution_;
  }

 private:
   void get_concurrency_degree (){
     if(const char* env_value = std::getenv("GRPPI_NUM_THREADS"))
     {
       try{
         int value = std::stoi(env_value);
         concurrency_degree_ = value;
       } catch(int e){
         std::cerr << "Non valid argument for number of threads" << std::endl;
       }
     }    
   }

   void get_ordering(){
     if(const char* env_value = std::getenv("GRPPI_ORDERING"))
     {
       if(strcmp(env_value,"TRUE") != 0) 
         ordering_ = false;
     }
   }

   void get_queue_size(){
     if(const char* env_value = std::getenv("GRPPI_QUEUE_SIZE"))
     {
       try{
         int value = std::stoi(env_value);
         queue_size_ = value;
       } catch (int e){
         std::cerr << "Non valid argument for queue size" << std::endl;
       }
     }
   }

   void get_queue_mode(){
     if(const char* env_value = std::getenv("GRPPI_QUEUE_MODE"))
     {
       if(strcmp(env_value,"lockfree") != 0) 
         queue_mode_ = queue_mode::lockfree;   
     }
   }
  
   void get_dynamic_execution(){
     if(const char* env_value = std::getenv("GRPPI_DYN_EXECUTION"))
     {
       dynamic_execution_ = env_value;
     }
   }

 public:
   configuration() :
    concurrency_degree_(static_cast<int>(std::thread::hardware_concurrency())),
    ordering_(true),
    queue_size_(100),
    queue_mode_(queue_mode::blocking),
    dynamic_execution_("seq")
  { 
    get_concurrency_degree();
    get_ordering();
    get_queue_size();
    get_queue_mode();
    get_dynamic_execution();
  }

private:
  int concurrency_degree_;
  bool ordering_;
  int queue_size_;
  queue_mode queue_mode_;
  std::string dynamic_execution_;

};

}

#endif
