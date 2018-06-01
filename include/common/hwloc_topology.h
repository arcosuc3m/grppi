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
#ifndef GRPPI_COMMON_HWLOC_TOPOLOGY_H
#define GRPPI_COMMON_HWLOC_TOPOLOGY_H

#include <thread>

#ifdef GRPPI_HWLOC

#include <hwloc.h>

namespace grppi {

class hwloc_processor_unit {
private:
  hwloc_processor_unit(const hwloc_obj_t & o) : object_{o} {}

public:
  int index() { return object_->logical_index; }

  int os_index() { return object_->os_index; }

  hwloc_processor_unit operator++(int) {
    hwloc_processor_unit old{*this};
    this->operator++();
    return old;
  }

  hwloc_processor_unit & operator++() {
    if (object_->next_cousin != nullptr) {
      object_ = object_->next_cousin;
    }
    else {
      while (object_->prev_cousin != nullptr) {
        object_ = object_->prev_cousin;
      }
    }
    return *this;
  }

private:
  hwloc_obj_t object_;

  friend class hwloc_topology;
};

class hwloc_topology {
public:

  hwloc_topology() {
    ::hwloc_topology_init(&topology_);
    ::hwloc_topology_load(topology_);

    int numa_depth = ::hwloc_get_type_or_below_depth(topology_, HWLOC_OBJ_NUMANODE);
    if (HWLOC_TYPE_DEPTH_UNKNOWN != numa_depth) {
      numa_nodes_ = ::hwloc_get_nbobjs_by_depth(topology_, numa_depth);
    }

    int core_depth = ::hwloc_get_type_or_below_depth(topology_, HWLOC_OBJ_CORE);
    if (HWLOC_TYPE_DEPTH_UNKNOWN != core_depth) {
      core_nodes_ = ::hwloc_get_nbobjs_by_depth(topology_, core_depth);
    }

    pu_depth_ = ::hwloc_get_type_or_below_depth(topology_, HWLOC_OBJ_PU);
    if (HWLOC_TYPE_DEPTH_UNKNOWN != pu_depth_) {
      pu_nodes_ = ::hwloc_get_nbobjs_by_depth(topology_, pu_depth_);
    }

  }

  ~hwloc_topology() {
    ::hwloc_topology_destroy(topology_);
  }

  hwloc_processor_unit first_processor_unit() const {
    int depth = ::hwloc_get_type_or_below_depth(topology_, HWLOC_OBJ_PU);
    hwloc_obj_t obj = hwloc_get_obj_by_depth(topology_, depth, 0);
    return {obj};
  }

  int numa_nodes() const { return numa_nodes_; }

  int core_nodes() const { return core_nodes_; }

  int logical_core_nodes() const { return pu_nodes_; }

  void pin_thread(std::thread & t, hwloc_processor_unit & pu) const {
    ::hwloc_set_thread_cpubind(topology_, t.native_handle(), 
        pu.object_->cpuset, 0);
#if HWLOC_API_VERSION>0x00020000
    ::hwloc_set_membind(topology_, pu.object_->nodeset, 
        HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
#else
    ::hwloc_set_membind(topology_, pu.object_->nodeset, 
        HWLOC_MEMBIND_BIND, 0);
#endif
  }

private:
  hwloc_topology_t topology_;
  int numa_nodes_ = 0;
  int core_nodes_ = 0;
  int pu_depth_ = 0;
  int pu_nodes_ = 0;
};

}

#endif // GRPPI_HWLOC


#endif
