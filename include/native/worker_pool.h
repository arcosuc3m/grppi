/**
* @version		GrPPI v0.3
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef GRPPI_NATIVE_WORKER_POOL_H
#define GRPPI_NATIVE_WORKER_POOL_H

#include <thread>

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
    worker_pool(int num_threads) noexcept : num_threads_{num_threads} {}

    /**
    \brief Destructs the worker pool after joining with all threads in the 
    pool.
    */
    ~worker_pool() noexcept { wait(); }

    worker_pool(worker_pool &&) noexcept = default;
    worker_pool & operator=(worker_pool &&) noexcept = default;
    
    /**
    \brief Launch a function in the pool.
    \pre Number of running threads must be lower than number of pool threads.
    \tparam E Execution policy type.
    \tparam F Type for launched function.
    \tparam Args Type for launched function arguments.
    \param ex Execution policy.
    \param f Function to be launched.
    \param args Arguments for launched function.
    */
    template <typename E, typename F, typename ... Args>
    void launch(const E & ex, F f, Args && ... args) {
      // TODO: Precondition
      if (num_threads_ <= workers_.size()) 
        throw std::runtime_error{"Too many threads in worker"};

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
