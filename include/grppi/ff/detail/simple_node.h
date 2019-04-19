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
#ifndef GRPPI_FF_DETAIL_SIMPLE_NODE_H
#define GRPPI_FF_DETAIL_SIMPLE_NODE_H

#include "fastflow_allocator.h"

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
    return new (ff_arena) Output{transform_op_(*p_item)};
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
    grppi::optional<Output> result{generate_op_()};
    if (result) {
      return new (ff_arena) Output{*result};
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
    operator delete(p_item, ff_arena);
    return GO_ON;
  }

private:
  Consumer consume_op_;
};


} // namespace detail_ff

} // namespace grppi

#endif
