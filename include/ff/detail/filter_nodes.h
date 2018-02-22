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
 \brief Constant to be returned by a node if current value is filtered out.
 */
constexpr size_t filtered_value = ff::FF_EOS - 0x11;

template <typename Item, typename Predicate>
class ordered_filter_worker : public ff::ff_node_t<Item> {
public:
  ordered_filter_worker(Predicate && predicate) :
      predicate_{predicate}
  {}

  Item * svc(Item * p_item) {
    if (predicate_(*p_item)) {
      return p_item;
    }
    else {
      p_item->~Item();
      ff::ff_free(p_item);
      return reinterpret_cast<Item*>(filtered_value);
    }
  }

private:
  Predicate predicate_;
};

template <typename Item>
class ordered_filter_collector : public ff::ff_node_t<Item> {
public:
  ordered_filter_collector() = default;

  Item * svc(Item * p_item) {
    if (p_item == reinterpret_cast<Item*>(filtered_value)) {
      return this->GO_ON;
    }
    else {
      return p_item;
    }
  }
};

template <typename Item, typename Predicate>
class unordered_filter_worker : public ff::ff_node_t<Item> {
public:
  unordered_filter_worker(Predicate && predicate) :
      predicate_{predicate}
  {}

  Item * svc(Item * p_item) {
    if (predicate_(*p_item)) {
      return p_item;
    }
    else {
      p_item->~Item();
      ff::ff_free(p_item);
      return reinterpret_cast<Item*>(filtered_value);
    }
  }

private:
  Predicate predicate_;
};


template <typename Item>
class unordered_filter_collector : public ff::ff_node_t<Item> {
public:
  unordered_filter_collector() = default;

  Item * svc(Item * p_item) {
    static_assert(sizeof(std::size_t) == sizeof(std::uintptr_t));
    if (p_item == reinterpret_cast<Item*>(filtered_value)) {
      return this->GO_ON;
    }
    else {
      return p_item;
    }
  }
};
template <typename Item>
class unordered_filter_emitter : public ff::ff_node_t<Item> {
public:
  unordered_filter_emitter() = default;

  Item * svc(Item * p_item) {
    return p_item;
  }
};



} // namespace detail_ff

} // namespace grppi

#endif
