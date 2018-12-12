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

enum class execution_backend {
  seq,
  native,
  omp,
  tbb,
  ff,
  task
};

class environment_option_getter {
public:
  char const * operator()(char const * var_name) { return std::getenv(var_name); }
};

template <typename OptionGetter = environment_option_getter>
class configuration {
public:

  configuration() {
    OptionGetter option_getter;
    set_concurrency_degree(option_getter("GRPPI_NUM_THREADS"));
    set_ordering(option_getter("GRPPI_ORDERING"));
    set_queue_size(option_getter("GRPPI_QUEUE_SIZE"));
    set_queue_mode(option_getter("GRPPI_QUEUE_MODE"));
    set_dynamic_backend(option_getter("GRPPI_DYN_BACKEND"));
  }

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

  execution_backend dynamic_backend() const noexcept {
    return dynamic_backend_;
  }

 private:

   void set_concurrency_degree(char const * str) noexcept {
     if (!str) return;
     try {
       int d = std::stoi(str);
       if (d<=0) {
         std::cerr << "GrPPI: Invalid concurrency degree \"" << d << "\"\n";
         return;
       }
       concurrency_degree_ = d;
     }
     catch (...) {
       std::cerr << "GrPPI: Invalid concurrency degree \"" << str << "\"\n";
     }
   }

   void set_ordering(char const * str) noexcept {
     if (!str) return;
     if (std::strcmp(str, "ordered") == 0) {
       ordering_ = true;
     }
     else if (std::strcmp(str, "unordered") == 0)  {
       ordering_ = false;
     }
     else {
       std::cerr << "GrPPI: Invalid ordering \"" << str << "\"\n";
     }
   }

  void set_queue_size(char const * str) noexcept {
    if (!str) return;
    try {
      int sz = std::stoi(str);
      if (sz <= 0) {
        std::cerr << "GrPPI: Invalid queue size  \"" << sz << "\"\n";
        return;
      }
      queue_size_ = sz;
    }
    catch (...) {
      std::cerr << "GrPPI: Invalid queue size  \"" << str << "\"\n";
    }  
  }

   void set_queue_mode(char const * str) noexcept {
     if (!str) return;
     if (strcmp(str, "blocking") == 0) {
       queue_mode_ = queue_mode::blocking;
     }
     else if (strcmp(str, "lockfree") == 0) {
       queue_mode_ = queue_mode::lockfree;
     }
     else {
       std::cerr << "GrPPI: Invalid queue mode \"" << str << "\"\n";
     }
   }
  
   void set_dynamic_backend(char const * str) noexcept {
     if (!str) return;
     if (std::strcmp("seq", str) == 0) {
       dynamic_backend_ = execution_backend::seq;
     }
     else if (std::strcmp("native", str) == 0) {
       dynamic_backend_ = execution_backend::native;
     }
     else if (std::strcmp ("omp", str) == 0) {
       dynamic_backend_ = execution_backend::omp;
     }
     else if (std::strcmp("tbb", str) == 0) {
       dynamic_backend_ = execution_backend::tbb;
     }
     else if (std::strcmp("ff", str) == 0) {
       dynamic_backend_ = execution_backend::ff;
     }
     else if (std::strcmp("task", str) == 0) {
       dynamic_backend_ = execution_backend::task;
     }
     else {
       std::cerr << "GrPPI: Invalid backend \"" << str << "\"\n";
     }
   }


public:

  constexpr static int default_queue_size = 100;

private:
  int concurrency_degree_ = static_cast<int>(std::thread::hardware_concurrency());
  bool ordering_ = true;
  int queue_size_ = default_queue_size;
  queue_mode queue_mode_ = queue_mode::blocking;
  execution_backend dynamic_backend_ = execution_backend::seq;

};

template <typename OptionGetter>
constexpr int configuration<OptionGetter>::default_queue_size;

}

#endif
