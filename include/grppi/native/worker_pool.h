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
#ifndef GRPPI_NATIVE_WORKER_POOL_H
#define GRPPI_NATIVE_WORKER_POOL_H

#include <thread>
#include <vector>

namespace grppi {

/**
\brief Pool of worker threads.
This class offers a simple pool of worker threads.
\note Current version does not support more threads than the initially set
number of threads.
*/
class worker_pool {
  public:

    /**
    \brief Creates a worker pool with a number of threads.
    \param num_threads Number of threads for the pool.
    */
    worker_pool(int num_threads) noexcept : 
        num_threads_{num_threads},
        workers_{}
    {}

    /**
    \brief Destructs the worker pool after joining with all threads in the 
    pool.
    */
    ~worker_pool() noexcept { this->wait(); }

    worker_pool(worker_pool &&) noexcept = default;
    worker_pool & operator=(worker_pool &&) noexcept = default;
    
    /**
    \brief Launch a function in the pool.
    \tparam E Execution policy type.
    \tparam F Type for launched function.
    \tparam Args Type for launched function arguments.
    \param ex Execution policy.
    \param f Function to be launched.
    \param args Arguments for launched function.
    */
    template <typename E, typename F, typename ... Args>
    void launch(const E & ex, F f, Args && ... args) {
      workers_.emplace_back([=,&ex]() {
        auto manager = ex.thread_manager();
        f(args...);
      });
    }

    template <typename E, typename F, typename ... Args>
    void launch_tasks(const E & ex, F && f, Args && ... args) {
      for (int i=0; i<num_threads_; ++i) {
        workers_.emplace_back([=,&ex]() {
          auto manager = ex.thread_manager();
          f(args...);
        });
      }
    }

    /**
    \brief Wait until all launched tasks have been completed.
    \post Number of workers is 0.
    */
    void wait() noexcept {
      for (auto && w : workers_) { w.join(); }
      workers_.clear();
    }

  private:
    const int num_threads_;
    std::vector<std::thread> workers_;
};

}

#endif
