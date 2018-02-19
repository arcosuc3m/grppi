/**
* @version    GrPPI v0.3
* @copyright    Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license    GNU/GPL, see LICENSE.txt
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

#ifndef GRPPI_FF_DETAIL_ORDERED_STREAM_REDUCE_H
#define GRPPI_FF_DETAIL_ORDERED_STREAM_REDUCE_H

#ifdef GRPPI_FF

#include "../../reduce.h"

#include <ff/farm.hpp>
#include <ff/allocator.hpp>

#include "reduce_task.h"

namespace grppi {

namespace detail_ff {

/**
 * Reduce task.
 * This is the reduce actual task for FastFlow.
 */

/**
 * Reduce emitter.
 */
template <typename Item, typename Reducer>
class ordered_reduce_emitter : public ff::ff_node {
public:
  ordered_reduce_emitter(int window_size, int offset);
  void * svc(void * p_value);

private:
  int window_size_;
  int offset_;
  int skip_;
  std::vector<Item> items_;
};

template <typename Item, typename Reducer>
ordered_reduce_emitter<Item,Reducer>::ordered_reduce_emitter(int window_size, int offset) :
    window_size_{window_size},
    offset_{offset},
    skip_{-1},
    items_{}
{
    items_.reserve(window_size);
}

template <typename Item, typename Reducer>
void * ordered_reduce_emitter<Item,Reducer>::svc(void * p_value) {
  Item * p_item = static_cast<Item*>(p_value);

  if(items_.size() != window_size_)
    items_.push_back(*p_item);

  if(items_.size() == window_size_) {
    if(offset_ < window_size_) {
      ff_send_out(new reduce_task<Item>(items_));
      items_.erase(items_.begin(), std::next(items_.begin(), offset_));
      p_item->~Item();
      ::ff::ff_free(p_item);
      return GO_ON;
    }
    if (offset_ == window_size_) {
      ff_send_out(new reduce_task<Item>(items_));
      items_.erase(items_.begin(), items_.end());
      ::ff::ff_free(p_item);
      return GO_ON;
    } 
    else {
      if (skip_==-1) {
        ff_send_out(new reduce_task<Item>(items_));
        skip_++;
      } 
      else if (skip_ == (offset_-window_size_)) {
        skip_ = -1;
        items_.clear();
        items_.push_back( std::forward<Item>(*p_item) );
      } 
      else {
        skip_++;
      }
      p_item->~Item();
      ::ff::ff_free(p_item);
      return GO_ON;
    }
  } 
  else {
    p_item->~Item();
    ::ff::ff_free(p_item);
    return GO_ON;
  }
}

/**
 * Reduce worker.
 */
template <typename Item, typename Combiner>
class ordered_reduce_worker : public ff::ff_node {
public:

  ordered_reduce_worker(Combiner && combine_op) : combine_op_{combine_op} {}
  void * svc(void * p_value);

private:
  Combiner combine_op_;
};

template <typename Item, typename Combiner>
void * ordered_reduce_worker<Item,Combiner>::svc(void * p_value) {
  reduce_task<Item> * p_task = static_cast<reduce_task<Item>*>(p_value);
  void * p_result_buf = ::ff::ff_malloc(sizeof(Item));
  Item * p_result = new (p_result_buf) Item;
  Item identity{};

  constexpr ::grppi::sequential_execution seq{};
  *p_result = ::grppi::reduce(seq, p_task->values_.begin(), p_task->values_.end(),
      identity, combine_op_);

  delete p_task;
  return p_result_buf;
}


/**
 * Ordered stream reduce for FastFlow.
 */
template <typename TSin, typename Reducer, typename CombineOp>
class ordered_stream_reduce : public ff::ff_ofarm {
public:
  ordered_stream_reduce(Reducer && red_obj, int num_workers = 1);

  ~ordered_stream_reduce() {
    delete p_emitter_;
  }

private:
  int concurrency_degree_;
  std::vector<ff_node*> workers_;

  ordered_reduce_emitter<TSin,Reducer> * p_emitter_;
};

template <typename Item, typename Reducer, typename Combiner>
ordered_stream_reduce<Item,Reducer,Combiner>::ordered_stream_reduce(
    Reducer && red_obj, 
    int num_workers) 
:
    ff::ff_ofarm{false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, num_workers},
    concurrency_degree_{num_workers},
    workers_{},
    p_emitter_{nullptr} 
{
  for(int i=0; i<num_workers; ++i) {
    ordered_reduce_worker<Item,Combiner> * p_worker =
        new ordered_reduce_worker<Item,Combiner>{red_obj.combiner()};
    workers_.push_back(p_worker);
  }

  p_emitter_ = new ordered_reduce_emitter<Item,Reducer>{
      red_obj.window_size(), red_obj.offset()};
  add_workers(workers_);
  setEmitterF(p_emitter_);
}


} // namespace detail_ff

} // namespace grppi

#else

#endif // GRPPI_FF

#endif
