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
#include "ff_node_wrap.hpp"


namespace grppi {

template <typename InputIt, typename Transformer, typename Identity, typename Combiner>
Identity map_reduce ( parallel_execution_ff& ex, InputIt first, InputIt last,
		Identity identity, Transformer && transform_op, Combiner && combine_op) {

	bool notnested = ff::outer_ff_pattern;
	ff::outer_ff_pattern=false;
	auto nw = ex.concurrency_degree();

	ssize_t total_parfor_size = last-first;
	ff::ParallelForPipeReduce<Identity> pfr(nw, true);

	pfr.parallel_reduce_idx(0,total_parfor_size,1,total_parfor_size/nw,
			// Map
			[&](const long internal_start, const long internal_stop, const int thid) {
		for (size_t internal_it =internal_start; internal_it < internal_stop; ++internal_it)
			*(firstOut+internal_it) = transform_op( *(first+internal_it) );
	},
			// Reduce
			[&]( ) {

	});

}



} //namespace
#endif


#endif /* GRPPI_FF_MAPREDUCE_H_ */
