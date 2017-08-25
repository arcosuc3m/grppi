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

/**
\addtogroup reduce_pattern
@{
\addtogroup reduce_pattern_ff FastFlow parallel reduce pattern
\brief FF parallel implementation of the \ref md_reduce.
@{
*/

/**
\brief Invoke \ref md_reduce with identity value
on a data sequence with parallel FF execution.
\tparam InputIt Iterator type used for input sequence.
\tparam Identity Type for the identity value.
\tparam Combiner Callable type for the combiner operation.
\param ex Parallel native execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param identity Identity value for the combiner operation.
\param combiner_op Combiner operation for the reduction.
*/
template < typename InputIt, typename Identity, typename Combiner>
auto reduce(parallel_execution_ff & ex,
            InputIt first, InputIt last, 
            Identity & identity,
            Combiner && combine_op)
{

	ssize_t total_parfor_size = last-first;
	ff::ParallelForReduce<Identity> pfr;

	Identity vaR = identity;
	Identity vaR_identity = identity;

	auto final_red = [&](typename std::iterator_traits<InputIt>::value_type a,
			typename std::iterator_traits<InputIt>::value_type r) {
		vaR = combine_op(a, r);
	};

	pfr.parallel_reduce(vaR, vaR_identity, 0, total_parfor_size, 1, total_parfor_size/ex.concurrency_degree(),
			[&](const long internal_it, typename std::iterator_traits<InputIt>::value_type &vaR) {
		vaR = combine_op( vaR, *(first+internal_it) );
	}, final_red, ex.concurrency_degree());

	//std::cout << "[REDUCE_FF] vaR: " << vaR << " identity: " << vaR_identity << std::endl;

	return vaR;

   
}

/**
@}
@}
*/

}

#endif /* GRPPI_FF */





#endif /* GRPPI_FF_REDUCE_H */
