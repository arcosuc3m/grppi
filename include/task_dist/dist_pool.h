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
#ifndef GRPPI_DIST_POOL_H
#define GRPPI_DIST_POOL_H

#include <future>
#include <iostream>
#include <vector>
#include "../common/mpmc_queue.h"
#include <functional>
#include <atomic>
#include <map>
#include <tuple>
#include <memory>

#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace grppi{

template <typename Scheduler>
class dist_pool
{
  public:
    using task_type = typename Scheduler::task_type;

    //dist_pool() = delete;
 
    void init (Scheduler * sched, long pool_size)
    {
      scheduler = sched;
      COUT << "dist_pool::dist_pool pool_size = " << pool_size << ENDL;
      for (auto i=0; i<pool_size; i++){
         pool_threads.emplace_back(std::thread(
              [this, i](){
                try {
                COUT << "dist_pool::dist_pool (" << i << "): thread begin" << ENDL;
                auto t = task_type{};
                while(1){
                  t = scheduler->get_task(t);
                  if( t == task_type{})
                    break;
                  COUT << "dist_pool::dist_pool (" << i << "): exec begin: task = (" << t.get_id()
                     << ", " << t.get_task_id() << ")" << ENDL;
                  scheduler->functions[t.get_id()](t);
                  COUT << "dist_pool::dist_pool (" << i << "): exec end: task = (" << t.get_id()
                     << ", " << t.get_task_id() << ")" << ENDL;
                }
                COUT << "dist_pool::dist_pool (" << i << "): thread end " << ENDL;
                } catch(const std::exception &e) {
                  std::cerr << "dist_pool - pool_threads " << e.what() << '\n';
                }
              }
         ));
       }
    }
   
    //dist_pool(const dist_pool &) = delete;
 
    //dist_pool(const dist_pool&&) = delete;

    void __attribute__ ((noinline)) finalize_pool()
    {
       COUT << "dist_pool::finalize_pool BEGIN" << ENDL;

       for(long i=0; i < pool_threads.size(); i++){
          pool_threads[i].join();
       }
       pool_threads.clear();
      COUT << "dist_pool::finalize_pool END" << ENDL;

    }

  private: 
    Scheduler * scheduler = NULL;
//    std::vector<std::future<void>> pool_threads;
    std::vector<std::thread> pool_threads;
};

}
#endif
