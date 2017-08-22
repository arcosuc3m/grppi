/*
 * reduce.h
 *
 *  Created on: Aug 22, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_REDUCE_H
#define GRPPI_FF_REDUCE_H

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"

#include <ff/node.hpp>
#include <ff/parallel_for.hpp>
#include <ff/allocator.hpp>
#include "ff_node_wrap.hpp"

namespace grppi {

template < typename InputIt, typename Identity, typename Combiner>
auto reduce(parallel_execution_ff & ex,
            InputIt first, InputIt last, 
            Identity identity,
            Combiner && combine_op)
{

	ssize_t total_parfor_size = last-first;
	ff::ParallelForReduce<Identity> pfr;
	Identity vaR; //, vaR_identity;

	pfr.parallel_reduce(vaR, identity, first, last, 1, total_parfor_size/ex.concurrency_degree(), combine_op,
			combine_op, ex.concurrency_degree());

	//std::cout << "[REDUCE_FF] vaR: " << vaR << " identity: " << identity << std::endl;

	return vaR;

   
}

/**
@}
@}
*/

}

#endif





#endif /* GRPPI_FF_REDUCE_H */
