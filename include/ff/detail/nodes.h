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

#ifndef GRPPI_FF_DETAIL_SIMPLE_NODE_H
#define GRPPI_FF_DETAIL_SIMPLE_NODE_H

#include "fastflow_allocator.h"
#include "reduce_task.h"

#include "../../reduce.h"

#include <ff/allocator.hpp>
#include <ff/node.hpp>

namespace grppi {

namespace detail_ff {

/**
\brief Fastflow node for a pipeline transformation stage.
\tparam Input Data type for the input value.
\tparam Output Data type for the output value.
\tparam Transformer Callable type for a transformation.
*/
template <typename Input, typename Output, typename Transformer>
class node_impl : public ff::ff_node_t<Input,Output> {
public:

  node_impl(Transformer && transform_op) : 
      transform_op_{transform_op}
  {}

  Output * svc(Input * p_item) {
    return fastflow_allocator<Output>::allocate(transform_op_(*p_item));
  } 

private:
  Transformer transform_op_;
};

/**
\brief Fastflow node for a pipeline generation stage.
\tparam Output Data type for the output value.
\tparam Generator Callable type for a generator.
*/
template <typename Output, typename Generator>
class node_impl<void,Output,Generator> : public ff::ff_node {
public:

  node_impl(Generator && generate_op) :
      generate_op_{generate_op}
  {}

  void * svc(void *) {
    std::experimental::optional<Output> result{generate_op_()};
    if (result) {
      return fastflow_allocator<Output>::allocate(*result);
    }
    else {
      return EOS;
    }
  }

private:
  Generator generate_op_;
};

/**
\brief Fastflow node for a pipeline consumer stage.
\tparam Input Data type for the input value.
\tparam Consumer Callable type for a consumer.
*/
template <typename Input, typename Consumer>
class node_impl<Input,void,Consumer> : public ff::ff_node_t<Input,void> {
public:

  node_impl(Consumer && consume_op) :
      consume_op_{consume_op}
  {}

  void * svc(Input * p_item) {
    consume_op_(*p_item);
    fastflow_allocator<Input>::deallocate(p_item);
    return GO_ON;
  }

private:
  Consumer consume_op_;
};

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

template<typename Item, typename Combiner>
class unordered_reduce_emitter : public ff::ff_node {
public:
  unordered_reduce_emitter(int window_size, int offset) :
      window_size_{window_size},
      offset_{offset},
      skip_{-1},
      items_{}
  {
    items_.reserve(window_size_);
  }

  void *svc(void *t); 

private:
  int window_size_;
  int offset_;
  int skip_;
  std::vector<Item> items_;
};

template<typename Item, typename Combiner>
void * unordered_reduce_emitter<Item,Combiner>::svc(void * p_value) {
  Item * p_item = static_cast<Item*>(p_value);

  if(items_.size() != window_size_) {
    items_.push_back(*p_item);
  }

  if(items_.size() == window_size_) {
    if(offset_ < window_size_) {
      this->ff_send_out(new reduce_task<Item>{items_});
      items_.erase(items_.begin(), std::next(items_.begin(), offset_));
      p_item->~Item();
      ::ff::ff_free(p_item);
      return GO_ON;
    }
    if (offset_ == window_size_) {
      this->ff_send_out(new reduce_task<Item>{items_});
      items_.erase(items_.begin(), items_.end());
      p_item->~Item();
      ::ff::ff_free(p_item);
      return GO_ON;
    } 
    else {
      if(skip_ == -1) {
        this->ff_send_out( new reduce_task<Item>(items_) );
        skip_++;
      } 
      else if(skip_ == (offset_ - window_size_)) {
        skip_ = -1;
        items_.clear();
        items_.push_back(*p_item);
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

// -- stream-reduce workers
template<typename Item, typename Combiner>
class unordered_reduce_worker : public ff::ff_node {
public:
  unordered_reduce_worker(Combiner && combiner) : 
      combiner_{std::move(combiner)}
  {}

  void *svc(void * p_value) {
    reduce_task<Item> * p_task = static_cast<reduce_task<Item>*>(p_value);

    void * p_out_item = ::ff::ff_malloc(sizeof(Item));
    Item * p_result = new (p_out_item) Item{};
    Item identity{};

    constexpr ::grppi::sequential_execution seq{};
    *p_result = ::grppi::reduce(seq, p_task->values_.begin(), p_task->values_.end(),
		identity, combiner_);

    delete p_task;
    return p_out_item;
  }

private:
  Combiner combiner_;
};


class unordered_reduce_collector : public ff::ff_node {
public:
  unordered_reduce_collector() = default;

  void * svc(void * p_value) { return p_value; }
};

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
