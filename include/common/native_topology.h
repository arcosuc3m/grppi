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
#ifndef GRPPI_COMMON_NATIVE_TOPOLOGY_H
#define GRPPI_COMMON_NATIVE_TOPOLOGY_H

#include <thread>

namespace grppi {

class native_processor_unit {
private:
  native_processor_unit(int idx, int nproc) : index_{idx}, max_{nproc} {}

public:
  int index() { return index_; }

  int os_index() { return index_; }

  native_processor_unit & operator++() {
    index_++;
    return *this;
  }

  native_processor_unit operator++(int) {
    native_processor_unit old{*this};
    index_ = (index_+1) % max_;
    return old;
  }

private:
  int index_;
  int const max_;

  friend class native_topology;
};

class native_topology {
public:

  native_topology() : 
      numa_nodes_{1},
      core_nodes_{static_cast<int>(std::thread::hardware_concurrency())},
      pu_nodes_{core_nodes_}
  {}

  native_processor_unit first_processor_unit() const {
    return {0, pu_nodes_};
  }

  int numa_nodes() const { return numa_nodes_; }

  int core_nodes() const { return core_nodes_; }

  int logical_core_nodes() const { return pu_nodes_; }

  void pin_thread(std::thread &, hwloc_processor_unit &) const {}

private:
  int numa_nodes_ = 0;
  int core_nodes_ = 0;
  int pu_nodes_ = 0;
};

}

#endif
