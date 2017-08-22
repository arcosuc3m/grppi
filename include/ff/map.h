/*
 * map.h
 *
 *  Created on: Aug 4, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_MAP_H_
#define GRPPI_FF_MAP_H_

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"
#include <ff/node.hpp>
#include <ff/parallel_for.hpp>
#include <ff/allocator.hpp>
#include "ff_node_wrap.hpp"


using namespace std;
namespace grppi {

template <typename InputIt, typename OutputIt, typename Transformer>
inline void map(parallel_execution_ff & ex,
		InputIt first,
		InputIt last,
		OutputIt firstOut,
		Transformer const & taskf){

	ssize_t total_parfor_size = last-first;
	ff::ParallelFor pf;

	pf.parallel_for_idx(0,total_parfor_size,1,total_parfor_size/ex.concurrency_degree(),
			[&](const long internal_ff_start,const long internal_ff_stop,const int internal_ff_thid) {
		for (size_t internal_ff_it = internal_ff_start; internal_ff_it < internal_ff_stop; ++internal_ff_it)
			*(firstOut+internal_ff_it) = taskf(*(first+internal_ff_it));
	},ex.concurrency_degree());
}

// TODO: Variadic Template not properly working in FF parallel_for (fabio)
template <typename InputIt, typename OutputIt,  typename Transformer, typename ... MoreIn>
inline void map(parallel_execution_ff & ex,
		InputIt first,
		InputIt last,
		OutputIt firstOut,
		Transformer const & taskf,
		MoreIn ... inputs ){
	//bool notnested = ff::outer_ff_pattern;
	//ff::outer_ff_pattern=false;

	ssize_t total_parfor_size = last-first;
	ff::ParallelFor pf;

	pf.parallel_for_idx(0,total_parfor_size,1,total_parfor_size/ex.concurrency_degree(),
			[&](const long internal_ff_start,const long internal_ff_stop,const int internal_ff_thid){
		//std::cout << "Th " << internal_ff_thid << " from " << internal_ff_start << " to " << internal_ff_stop << "\n";
		for (size_t internal_ff_it = internal_ff_start; internal_ff_it < internal_ff_stop; ++internal_ff_it)
			*(firstOut+internal_ff_it) = taskf(*(first+internal_ff_it), *inputs...);
	},ex.concurrency_degree());
}



#endif



#endif /* GRPPI_FF_MAP_H_ */
