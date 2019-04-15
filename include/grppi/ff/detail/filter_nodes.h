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
#ifndef GRPPI_FF_DETAIL_FILTER_NODES_H
#define GRPPI_FF_DETAIL_FILTER_NODES_H

#include "fastflow_allocator.h"

#include <ff/node.hpp>

namespace grppi {

namespace detail_ff {

/**
 \brief Get a pointer representation of the filter constant to be used
        as special value when a value is filtered out.
 */
template <typename T>
constexpr T * filtered_value() { 
  static_assert(sizeof(std::size_t) == sizeof(std::uintptr_t),
     "std::size_t and pointers have different sizes");
  return reinterpret_cast<T*>(std::size_t(ff::FF_EOS - 0x11));
}

/**
 \brief Worker that passes a value to next stage if the predicate is satisfied
        or the filtered_value constant otherwise.
 */
template <typename Item, typename Predicate>
class filter_worker : public ff::ff_node_t<Item> {
public:
  filter_worker(Predicate && predicate) :
      predicate_{std::forward<Predicate>(predicate)}
  {}

  Item * svc(Item * p_item) {
    if (predicate_(*p_item)) {
      return p_item;
    }
    else {
      operator delete(p_item,ff_arena);
      return filtered_value<Item>();
    }
  }

private:
  Predicate predicate_;
};

/**
 \brief Collector node for a filter.
 */
template <typename Item>
class filter_collector : public ff::ff_node_t<Item> {
public:
  filter_collector() = default;

  Item * svc(Item * p_item) {
    if (p_item == filtered_value<Item>()) {
      return this->GO_ON;
    }
    else {
      return p_item;
    }
  }
};

/**
 \brief Emitter for a filter stage.
 */
template <typename Item>
class filter_emitter : public ff::ff_node_t<Item> {
public:
  filter_emitter() = default;

  Item * svc(Item * p_item) { return p_item; }
};



} // namespace detail_ff

} // namespace grppi

#endif
