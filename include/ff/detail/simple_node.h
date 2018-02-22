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


} // namespace detail_ff

} // namespace grppi

#endif
