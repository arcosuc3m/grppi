/**
* @version		GrPPI v0.1
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

#ifndef PPI_COMMON
#define PPI_COMMON

#include <thread>
#include <atomic>

#include "mpmc_queue.h"

#ifdef GRPPI_THRUST
        #include <thrust/execution_policy.h>
        #include <thrust/system/omp/execution_policy.h>

        #ifdef GRPPI_TBB
                #include <thrust/system/tbb/execution_policy.h>
        #endif
#endif
#include "seq_policy.h"
#include "thread_policy.h"

#ifdef GRPPI_OMP
#include "omp_policy.h"
#endif

#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

namespace grppi{

/** @brief Set the execution mode to sequencial */
struct sequential_execution {
  bool ordering = true;
  int num_threads=1;
  bool lockfree = false;
  /** @brief set num_threads to 1 in order to sequential execution */
  sequential_execution(){};
};

//extern bool initialised;
//extern thread_pool pool;
/** @brief Set the execution mode to parallel with posix thread implementation 
 */
struct parallel_execution_thr{
  public: 
  thread_pool pool;
  bool ordering = true;
  int num_threads = 4;
  bool lockfree = false;

  int get_threadID(){
      while (lock.test_and_set(std::memory_order_acquire));
      auto it = std::find(thid_table.begin(), thid_table.end(), std::this_thread::get_id());
      auto id = std::distance(thid_table.begin(), it);
      lock.clear(std::memory_order_release);  
      return id; 
  }
  
  void register_thread(){
      while (lock.test_and_set(std::memory_order_acquire));
      thid_table.push_back(std::this_thread::get_id());    
      lock.clear(std::memory_order_release);  
  }
  
  void deregister_thread(){
      while (lock.test_and_set(std::memory_order_acquire));
      thid_table.erase(std::remove(thid_table.begin(), thid_table.end(),std::this_thread::get_id()), thid_table.end());
      lock.clear(std::memory_order_release);  
  }
  /** @brief Set num_threads to the maximum number of thread available by the
   *    hardware
   */
  parallel_execution_thr(){ /*if(!initialised)*/ pool.initialise(this->num_threads); };

  /** @brief Set num_threads to _threads in order to run in parallel
   *
   *  @param _threads number of threads used in the parallel mode
   */
  parallel_execution_thr(int _threads){ num_threads=_threads;  pool.initialise (_threads);};

  /** @brief Set num_threads to _threads in order to run in parallel and allows to disable the ordered execution
   *
   *  @param _threads number of threads used in the parallel mode
   *  @param _order enable or disable the ordered execution
   */
  parallel_execution_thr(int _threads, bool order){ num_threads=_threads; ordering = order; pool.initialise (_threads);};
  private: 
     std::atomic_flag lock = ATOMIC_FLAG_INIT;
     std::vector<std::thread::id> thid_table;

};


#ifdef GRPPI_OMP
/** @brief Set the execution mode to parallel with ompenmp framework 
 *    implementation 
 */
struct parallel_execution_omp{
  bool ordering = true;
  bool lockfree = false;
  int num_threads = 4;
  int get_threadID(){
     return omp_get_thread_num();
  }
  /** @brief Set num_threads to the maximum number of thread available by the
   *    hardware
   */
  parallel_execution_omp(){};

  /** @brief Set num_threads to _threads in order to run in parallel
   *
   *  @param _threads number of threads used in the parallel mode
   */
  parallel_execution_omp(int _threads){num_threads=_threads; };

  /** @brief Set num_threads to _threads in order to run in parallel and allows to disable the ordered execution
   *
   *  @param _threads number of threads used in the parallel mode
   *  @param _order enable or disable the ordered execution
   */
  parallel_execution_omp(int _threads, bool order){ num_threads=_threads; ordering = order;};

};

=======
#ifdef GRPPI_THRUST
#include "thrust_policy.h"
>>>>>>> fc0566fc3248a14aa97736debcdc013d365f2ee6
#endif

#ifdef GRPPI_TBB
#include "tbb_policy.h"
#endif

#include "optional.h"

#include "is_iterator.h"
#include "mpmc_queue.h"

namespace grppi{

template <typename T>
class _has_arguments
{
 template <typename C> static char test( typename  std::result_of<C()>::type*  );
 template <typename C> static long test( ... );
 public:
    static bool const value = !( sizeof( test<T>(0) ) == sizeof(char) );
};


template <typename InputIt>
void GetStart(int n, int tid, InputIt& in){
    in = in + (n*tid);
}

template <typename InputIt, typename ... MoreIn>
void GetStart(int n, int tid, InputIt& in, MoreIn ... inputs){
    in = in + (n*tid);
    GetStart(n,tid,inputs...);
}

//Update iterators
template <typename InputIt>
void NextInputs(InputIt &in){
   in++;
}

template <typename InputIt, typename ... MoreIn>
void NextInputs(InputIt &in, MoreIn ... inputs){
   in++;
   NextInputs(inputs...);
}


template <typename E,typename Stage, typename ... Stages>
class PipelineObj{
   public:
      E * exectype;
      std::tuple<Stage *, Stages *...> stages;
      PipelineObj(E &p, Stage s, Stages ... sts):stages(std::make_tuple(&s, &sts...)) { exectype = &p;}
};

template <typename E,class TaskFunc, class RedFunc>
class ReduceObj
{
   public:
      TaskFunc * task;
      RedFunc * red;
      E exectype;
      ReduceObj(E s, TaskFunc farm, RedFunc r){exectype=s; task = &farm; red= &r;}
};

template <typename E,class TaskFunc>
class FarmObj
{
   public:
      TaskFunc * task;
      E * exectype;
      int farmtype;
      FarmObj(E &s,TaskFunc f){exectype=&s; task = &f;};


};

template <typename E,class TaskFunc>
class FilterObj
{
   public:
      TaskFunc * task;
      E *exectype;
      int filtertype;
      FilterObj(E& s,TaskFunc f){exectype=&s; task = &f;};
};

}

#endif
