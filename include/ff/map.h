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
#include "ff_node_wrap.hpp"


using namespace std;
namespace grppi {

/**
\addtogroup map_pattern
@{
\addtogroup map_pattern_ff FF parallel \ref md_map pattern.
\brief FF parallel implementation of \ref md_map.
@{
*/

/**
\brief Invoke \ref md_map on a data sequence with FF
parallel execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Parallel FF execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transf_op Transformation operation.
*/
template <typename InputIt, typename OutputIt, typename Transformer>
inline void map(parallel_execution_ff & ex,
		InputIt first,
		InputIt last,
		OutputIt firstOut,
		Transformer const & taskf) {

	ssize_t total_parfor_size = last-first;
	ff::ParallelFor pf;

	pf.parallel_for_idx(0, total_parfor_size, 1, total_parfor_size/ex.concurrency_degree(),
			[&](const long internal_start, const long internal_stop, const int thid) {
		for (size_t internal_it = internal_start; internal_it < internal_stop; ++internal_it)
			*(firstOut+internal_it) = taskf( *(first+internal_it) );
	}, ex.concurrency_degree());
}


/**
\brief Invoke \ref md_map on a data sequence with FF
parallel execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\tparam OtherInputIts Iterator types used for additional input sequences.
\param ex Parallel FF execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transf_op Transformation operation.
\param more_firsts Additional iterators with first elements of additional sequences.
*/
template <typename InputIt, typename OutputIt,
		typename Transformer, typename ... OtherInputsIts>
inline void map(parallel_execution_ff & ex,
		InputIt first,
		InputIt last,
		OutputIt firstOut,
		Transformer const & taskf,
		OtherInputsIts ... more ) {

	ssize_t total_parfor_size = last-first;
	ff::ParallelFor pf;

	pf.parallel_for_idx(0, total_parfor_size, 1, total_parfor_size/ex.concurrency_degree(),
			[&](const long internal_start, const long internal_stop, const int thid) {
		for (size_t internal_it = internal_start; internal_it < internal_stop; ++internal_it)
			*(firstOut+internal_it) = taskf( *(first+internal_it), *(more+internal_it)... );
	}, ex.concurrency_degree());
}

}

#endif /* GRPPI_FF */



#endif /* GRPPI_FF_MAP_H_ */
