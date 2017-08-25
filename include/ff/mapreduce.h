/*
 * mapreduce.h
 *
 *  Created on: Aug 23, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_MAPREDUCE_H
#define GRPPI_FF_MAPREDUCE_H

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"

#include <ff/node.hpp>
#include <ff/parallel_for.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include "ff_node_wrap.hpp"


namespace grppi {

template <typename InputIt, typename Transformer, typename Identity, typename Combiner>
Identity map_reduce( parallel_execution_ff& ex, InputIt first, InputIt last,
		Identity identity, Transformer && transform_op, Combiner && combine_op) {

	size_t total_parfor_size = last-first;
	auto nw = ex.concurrency_degree();

	// FT: enable spinwait and spinbarrier- we use the same pattern twice
	// this constructor does not skip the warm-up phase
	ff::ParallelForReduce<Identity> pfr(nw, true, true);

	Identity vaR = identity;
	Identity var_id = identity;
	std::vector<Identity> partial_outs(total_parfor_size);

	// Map function
	auto Map = [&](const long internal_start, const long internal_stop, const int thid) {
		for (size_t internal_it = internal_start; internal_it < internal_stop; ++internal_it)
			partial_outs[internal_it] = transform_op(*(first+internal_it));
	};

	// Reduce funtion - this is the partial reduce function, executed in parallel
	auto Reduce = [&](const long internal_it, Identity &vaR) {
		vaR = combine_op( vaR, partial_outs[internal_it] );

	};

	// Final reduce - this reduces partial results and is executed sequentially
	auto final_red = [&](Identity a, Identity b) {
		vaR = combine_op(a, b);
	};

	// FT: consider implementing map+reduce as pipeline stages. could it be more efficient?
	pfr.parallel_for_idx(0, total_parfor_size, 1, total_parfor_size/ex.concurrency_degree(),
			Map, ex.concurrency_degree() );

	pfr.parallel_reduce(vaR, var_id, 0, total_parfor_size, 1, total_parfor_size/ex.concurrency_degree(),
			Reduce, final_red, ex.concurrency_degree());

	return vaR;
}



} //namespace
#endif


#endif /* GRPPI_FF_MAPREDUCE_H_ */
