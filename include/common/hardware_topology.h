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
#ifndef GRPPI_COMMON_HARDWARE_TOPOLOGY_H
#define GRPPI_COMMON_HARDWARE_TOPOLOGY_H

#include "../common/hwloc_topology.h"
#include "../common/native_topology.h"

namespace grppi {

class hardware_topology {

public:
#ifdef GRPPI_HWLOC
  using topology_type = hwloc_topology;
  using processor_type = hwloc_processor_unit;
#else
  using topology_type = native_topology;
  using processor_type = native_processor_unit;
#endif

public:

  hardware_topology() : topology_{} {}

  auto first_processor_unit() const {
    return topology_.first_processor_unit();
  }

  int numa_nodes() const { return topology_.numa_nodes(); }

  int core_nodes() const { return topology_.core_nodes(); }

  int logical_core_nodes() const { return topology_.logical_core_nodes(); }

  void pin_thread(std::thread & t, hwloc_processor_unit & pu) const {
    topology_.pin_thread(t,pu);
  }

private:
  topology_type topology_;
};

}

#endif
