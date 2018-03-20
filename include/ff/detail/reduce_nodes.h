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

#ifndef GRPPI_FF_DETAIL_REDUCE_NODES_H
#define GRPPI_FF_DETAIL_REDUCE_NODES_H

#include "fastflow_allocator.h"
#include "../../reduce.h"

#include <ff/allocator.hpp>
#include <ff/node.hpp>

namespace grppi {

namespace detail_ff {

/**
 \brief Reduce emitter.
 */
template <typename Item, typename Reducer>
class reduce_emitter : public ff::ff_node {
public:
  reduce_emitter(int window_size, int offset);
  void * svc(void * p_value);

private:
  void advance_large_offset(Item * p_item);
  void advance_small_offset();

private:
  int window_size_;
  int offset_;
  int skip_;
  std::vector<Item> items_;
};

template <typename Item, typename Reducer>
reduce_emitter<Item,Reducer>::reduce_emitter(int window_size, int offset) :
    window_size_{window_size},
    offset_{offset},
    skip_{-1},
    items_{}
{
    items_.reserve(window_size);
}

template <typename Item, typename Reducer>
void * reduce_emitter<Item,Reducer>::svc(void * p_value) 
{
  Item * p_item = static_cast<Item*>(p_value);

  if(static_cast<int>(items_.size()) != window_size_)
    items_.push_back(*p_item);

  if(static_cast<int>(items_.size()) == window_size_) {
    if(offset_ > window_size_) {
      advance_large_offset(p_item);
    }
    else {
      advance_small_offset();
    }
  } 

  operator delete(p_item, ff_arena);
  return GO_ON;
}

template <typename Item, typename Reducer>
void reduce_emitter<Item,Reducer>::advance_large_offset(Item * p_item) 
{
  if (skip_==-1) {
    auto * p_items_to_send = new std::vector<Item>{items_};
    ff_send_out(p_items_to_send);
    skip_++;
  } 
  else if (skip_ == (offset_ - window_size_)) {
    skip_ = -1;
    items_.clear();
    items_.push_back(*p_item);
  } 
  else {
    skip_++;
  }
}

template <typename Item, typename Reducer>
void reduce_emitter<Item,Reducer>::advance_small_offset() 
{
  auto * p_items_to_send = new std::vector<Item>{
      std::make_move_iterator(items_.begin()),
      std::make_move_iterator(items_.end())};
  auto it_last = (offset_ < window_size_) ?
      std::next(items_.begin(), offset_) :
      items_.end();
  items_.erase(items_.begin(), it_last);
  ff_send_out(p_items_to_send);
}

/**
 \brief Reduce worker.
 */
template <typename Item, typename Combiner>
class reduce_worker : public ff::ff_node {
public:

  reduce_worker(Combiner && combine_op) : combine_op_{combine_op} {}
  void * svc(void * p_value);

private:
  Combiner combine_op_;
};

template <typename Item, typename Combiner>
void * reduce_worker<Item,Combiner>::svc(void * p_value) {
  std::vector<Item> * p_items = static_cast<std::vector<Item>*>(p_value);

  Item identity{};
  constexpr ::grppi::sequential_execution seq{};
  Item * p_result = new (ff_arena) Item{
      ::grppi::reduce(seq, p_items->begin(), p_items->end(),
          identity, combine_op_)
  };

  delete p_items;
  return p_result;
}

/**
 \brief Reduce collector.
 */
class reduce_collector : public ff::ff_node {
public:
  reduce_collector() = default;

  void * svc(void * p_value) { return p_value; }
};


} // namespace detail_ff

} // namespace grppi

#endif
