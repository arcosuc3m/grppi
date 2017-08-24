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
Identity map_reduce ( parallel_execution_ff& ex, InputIt first, InputIt last,
		Identity identity, Transformer && transform_op, Combiner && combine_op) {

	size_t total_parfor_size = last-first;
	auto nw = ex.concurrency_degree();

	ff::ParallelForPipeReduce<std::vector<Identity>> pfr(nw, true);

	std::vector<Identity> R;
	R.reserve(nw);
	Identity vaR;

	auto Map = [&](const long internal_start, const long internal_stop, const int thid) {
		std::vector<Identity> partials;
		partials.reserve(internal_stop - internal_start);

		std::cout << "[MAPREDUCE] map stage" << std::endl;

		for (size_t internal_it = internal_start; internal_it < internal_stop; ++internal_it)
			partials[internal_it] = transform_op( *(first+internal_it) );

		ff::ff_node::ff_send_out(&partials);
	};

	auto Reduce = [&](std::vector<Identity> *p) {
		const std::vector<Identity> &parts = *p;

		std::cout << "[MAPREDUCE] reduce stage" << std::endl;

		for(size_t i=0; i<parts.size(); ++i)
			vaR = combine_op( vaR, parts[i] );

	};

	pfr.parallel_reduce_idx(0,total_parfor_size,1,total_parfor_size/nw, Map, Reduce);

	std::cout << "[MAPREDUCE] vaR: " << vaR << std::endl;

	return vaR;

}



} //namespace
#endif


#endif /* GRPPI_FF_MAPREDUCE_H_ */
