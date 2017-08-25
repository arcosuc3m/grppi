/*
 * pipeline.h
 *
 *  Created on: Aug 23, 2017
 *      Author: fabio
 */

#ifndef INCLUDE_FF_PIPELINE_H_
#define INCLUDE_FF_PIPELINE_H_

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"

#include <experimental/optional>

#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include <ff/allocator.hpp>
#include "ff_node_wrap.hpp"

namespace grppi {

// TODO
template <typename Generator, typename ... Transformers,
          requires_no_arguments<Generator> = 0>
void pipeline(parallel_execution_tbb & ex, Generator generate_op,
              Transformers && ... transform_ops) {

}

} // namespace

#endif // GRPPI_FF


#endif /* INCLUDE_FF_PIPELINE_H_ */
