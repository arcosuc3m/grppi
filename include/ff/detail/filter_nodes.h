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
 \brief Colletor node for a filter.
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
