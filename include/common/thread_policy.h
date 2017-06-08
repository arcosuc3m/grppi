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

#ifndef GRPPI_THREAD_POLICY_H
#define GRPPI_THREAD_POLICY_H

#include "../ppi_thr/pool.h"

#ifdef GRPPI_HWLOC
#include <hwloc.h>
#endif

#include <tuple>
#include <map>
#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>

namespace grppi {

//extern bool initialised;
//extern thread_pool pool;
/** @brief Set the execution mode to parallel with posix thread implementation 
 */
struct parallel_execution_thr {
  private:
   //NUMA and CPU affinity 
   #ifdef GRPPI_HWLOC
   hwloc_topology_t topo;
   std::map<int, hwloc_bitmap_t > cpu_bind;
   std::map<int, hwloc_bitmap_t > mem_bind;
   #endif
   //Number of threads, current active threads and their IDs
   int num_threads;
   int active_threads = 0;
   std::vector<std::thread::id> thid_table;
   std::atomic_flag lock = ATOMIC_FLAG_INIT;
   thread_pool pool;
   bool lockfree = false;
   bool ordering = false;

   //Apply the user defined CPU and NUMA affinity
   void apply_affinity(int tid)
   {
   #ifdef GRPPI_HWLOC
     auto thr_mask = cpu_bind.find(tid);
     if(cpu_bind.end() != thr_mask){
       hwloc_set_cpubind( topo, thr_mask->second, HWLOC_CPUBIND_THREAD );
     }
     auto mem_mask = mem_bind.find(tid);
     if(mem_bind.end() != mem_mask){
       hwloc_set_membind_nodeset(topo, mem_mask->second, HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
     }
   #endif
   };

 public:
   //Enables the output ordering and the use of lockfree queues
   void set_lockfree(bool value){ lockfree = value;}
   void set_ordered(bool value){ ordering = value;}
   
   bool is_lockfree(){ return lockfree;}
   bool is_ordered(){ return ordering;}
   
   //Create a new task for the thread pool 
   template <typename T>
   void create_task(T task){ pool.create_task(task);}

   //Register a new active thread
   void register_thread(){
     while (lock.test_and_set(std::memory_order_acquire));
     active_threads++;
     thid_table.push_back(std::this_thread::get_id());
     auto it = std::find(thid_table.begin(), thid_table.end(), std::this_thread::get_id());
     auto id = std::distance(thid_table.begin(), it);
     lock.clear(std::memory_order_release);
     apply_affinity(id);
   }

   /** @brief returns the ID of the current thread 
   */
   int get_threadID(){
     while (lock.test_and_set(std::memory_order_acquire));
     auto it = std::find(thid_table.begin(), thid_table.end(), std::this_thread::get_id());
     auto id = std::distance(thid_table.begin(), it);
     lock.clear(std::memory_order_release);
     return id;
   }

   //Deregister a thread
   void deregister_thread(){
     while (lock.test_and_set(std::memory_order_acquire));
     active_threads--;
     if(active_threads == 0) thid_table.clear();
     lock.clear(std::memory_order_release);
   }

   /** @brief Set num_threads to the maximum number of thread available by the
    *    hardware
    */
   parallel_execution_thr(){
   #ifdef GRPPI_HWLOC
     hwloc_topology_init(&topo);
     hwloc_topology_load(topo);
   #endif
     num_threads = std::thread::hardware_concurrency();
     pool.initialise(num_threads);
   };

   /** @brief Set num_threads to _threads in order to run in parallel
    *
    *  @param _threads number of threads used in the parallel mode
   */
   parallel_execution_thr(int _nthreads){
   #ifdef GRPPI_HWLOC
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
   #endif
    num_threads = _nthreads;
    pool.initialise(num_threads);
   };



   /** @brief Set num_threads to _threads in order to run in parallel and allows to disable the ordered execution
    *
    *  @param _threads number of threads used in the parallel mode
    *  @param _order enable or disable the ordered execution
    */
   parallel_execution_thr(int _nthreads, bool order){
   #ifdef GRPPI_HWLOC
     hwloc_topology_init(&topo);
     hwloc_topology_load(topo);
   #endif
     num_threads = _nthreads;
     pool.initialise(num_threads);
     ordering = order;
   };

   ~parallel_execution_thr(){
   #ifdef GRPPI_HWLOC
     for(auto it = cpu_bind.begin(); it != cpu_bind.end(); it++) hwloc_bitmap_free(it->second);
     for(auto it = mem_bind.begin(); it != mem_bind.end(); it++) hwloc_bitmap_free(it->second);
     hwloc_topology_destroy(topo);
   #endif
   };

   /** @brief Set num_threads to _threads to a new value
    *
    *  @param _threads number of threads 
    */
   void set_num_threads(int _nthreads){
   #ifdef GRPPI_HWLOC
     for(auto i = _nthreads; i< num_threads;i++){
       auto cit = cpu_bind.find(i);
       if(cit != cpu_bind.end()) hwloc_bitmap_free(cit->second);
       cpu_bind.erase(i);
       auto nit = mem_bind.find(i);
       if(nit != mem_bind.end()) hwloc_bitmap_free(nit->second);
       mem_bind.erase(i);
     }
   #endif
     num_threads = _nthreads;
   };
   /** @brief returns the number of threads
    */
   int get_num_threads() const{
     return num_threads;
   };
   /** @brief Set the memory affinity of a given thread to a set of NUMA nodes
    *
    *  @param tid ID of the thread that should be pinned
    *  @param nodeset vector of NUMA nodes IDs 
    */
   void set_numa_affinity(int tid, std::vector<int> nodeset){
   #ifdef GRPPI_HWLOC
     unsigned int length = nodeset.size();
     hwloc_bitmap_t numa_set = hwloc_bitmap_alloc();
     for(auto i = 0; i < length; i++){
       hwloc_bitmap_set( numa_set, nodeset[i] );
     }
     mem_bind[tid] = numa_set;
   #endif
   };
   /** @brief Set the CPU affinity of a given thread to a set of CPU nodes
    *
    *  @param tid ID of the thread that should be pinned
    *  @param cpuset vector of CPU IDs 
    */
   void set_thread_affinity(int tid, std::vector<int> cpuset){
   #ifdef GRPPI_HWLOC
     unsigned int length = cpuset.size();
     hwloc_bitmap_t cpu_set = hwloc_bitmap_alloc();
     for(auto i = 0; i < length; i++){
       hwloc_bitmap_set( cpu_set, cpuset[i] );
     }
     cpu_bind[tid] = cpu_set;
   #endif
   };
};
} // end namespace grppi


#endif
