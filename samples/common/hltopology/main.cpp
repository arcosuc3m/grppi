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
// Standard library
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

// grppi
#include "common/hwloc_topology.h"

std::mutex out_mutex;

void f(int k) {
  double aux = 0.0;
  for (int i=0; i<1000000; ++i) {
    aux += k*i;
  }
  std::lock_guard<std::mutex> l{out_mutex};
  std::cout << "Finishing thread " << k; 
  std::cout << " (" << std::this_thread::get_id() << ") ";
  std::cout << " at pu " << sched_getcpu() << "\n";
  std::cout << "  " << aux << "\n";
}
  
void print_nodes(const grppi::hwloc_topology & t) {
  std::cout << "Number of NUMA nodes: " << t.numa_nodes() << "\n";
  std::cout << "Number of core nodes: " << t.core_nodes() << "\n";
  std::cout << "Number of logical core nodes: " << t.logical_core_nodes() << "\n";

  grppi::hwloc_processor_unit p = t.first_processor_unit();
  int max = t.logical_core_nodes();
  std::cout << "Nodes (" << max << "):\n";
  for (int i=0; i<max; ++i) {
    std::cout << "Unit " << i << "<" << p.index() << "> -> (" 
        << p.os_index() << ")" << "\n";
    ++p;
  }
}

void run_threads(const grppi::hwloc_topology & t, int nthreads) {
  std::vector<std::thread> tasks;
  grppi::hwloc_processor_unit pu = t.first_processor_unit();
  for (int i=0; i<nthreads; ++i) {
    tasks.push_back(std::thread([=] { f(i); }));
    t.pin_thread(tasks[i], pu);
    {
      std::lock_guard<std::mutex> l{out_mutex};
      std::cout << "Pinned " << i << " with id " << tasks[i].native_handle()
          << " to pu " << pu.index() << " (" << pu.os_index() << ")" << "\n";
    }
    ++pu;
    if (!pu) pu = t.first_processor_unit();
  }

  for (auto && th : tasks) { th.join(); }
}

int main() {
    
  using namespace std;

  grppi::hwloc_topology t;
  print_nodes(t);
  run_threads(t,15);

  return 0;
}
