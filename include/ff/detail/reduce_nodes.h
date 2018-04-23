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
