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
#ifndef GRPPI_POOL_H
#define GRPPI_POOL_H

#include <future>
#include <iostream>
#include <vector>
#include "../common/mpmc_queue.h"
#include <functional>
#include <atomic>
#include <map>
#include <tuple>
#include <memory>

namespace grppi{

template <typename Scheduler>
class pool
{
  public:
    using task_type = typename Scheduler::task_type;

    pool() = delete;   
 
    pool(Scheduler &s,int pool_size):
    scheduler{s}
    {
      std::stringstream ss{};
      ss<<"\t\t";        
      for (auto i=0; i<pool_size; i++){
         std::string tabs = ss.str(); 
         pool_threads.emplace_back(std::thread(
              [this,tabs](){
                while(1){
                  task_type t = scheduler.get_task();
                  if( t == task_type{-1,-1})
                    break;
                  
                  print(tabs+" get task "+ std::to_string(t.get_task_id()));
                  print(tabs+" data.get("+std::to_string(t.get_data_location())+")");
                  print(tabs+" run() ");
                  scheduler.start_task(),
                  scheduler.functions[t.get_id()](t),
                  scheduler.finalize_task(t);
                  print(tabs+" data.set(...) ");
                  print(tabs+" end task ");
                }
              }
         ));
         ss<<"\t\t";        
       }
    }
   
    pool(const pool &) = delete;   
 
    pool(const pool&&) = delete;

    void __attribute__ ((noinline)) finalize_pool()
    {
       for(unsigned int i=0; i < pool_threads.size(); i++){
          scheduler.launch_task(task_type{-1,-1});
       }
       for(unsigned int i=0; i < pool_threads.size(); i++){
          pool_threads[i].join();
       }
       pool_threads.clear();
    }

    void launch_task(task_type t)
    {
      if( t != task_type{-1,-1} )
      {
        scheduler.launch_task(t);
      }
    }
 
    void end_seq(task_type t)
    {
      scheduler.notify_seq_end(t);
    }

    void seq_consume(task_type t)
    {
      scheduler.notify_consumer_end(t);
    }

  private: 
    Scheduler& scheduler;
//    std::vector<std::future<void>> pool_threads;
   std::vector<std::thread> pool_threads;
};

}
#endif
