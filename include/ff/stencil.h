/*
 * stencil.h
 *
 *  Created on: Aug 23, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_STENCIL_H
#define GRPPI_FF_STENCIL_H

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"
#include <ff/node.hpp>
#include <ff/parallel_for.hpp>
#include "ff_node_wrap.hpp"

namespace grppi {

/**
\addtogroup stencil_pattern
@{
\addtogroup stencil_pattern_ff FF Parallel stencil pattern
\brief FF parallel implementation of the \ref md_stencil.
@{
*/

/**
\brief Invoke \ref md_stencil on a data sequence with
TBB parallel execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\param ex FF parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation transform_operation.
\param neighbour_op Neighbourhood transform_operation.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer,
          typename Neighbourhood>
void stencil(parallel_execution_ff & ex,
             InputIt first, InputIt last, OutputIt first_out,
             StencilTransformer transform_op,
             Neighbourhood neighbour_op) {

	ssize_t total_parfor_size = last-first;
	ff::ParallelFor pf;

	pf.parallel_for_idx(0, total_parfor_size, 1, total_parfor_size/ex.concurrency_degree(),
			[&](const long internal_start, const long internal_stop, const int thid) {
		for (size_t internal_it = internal_start; internal_it < internal_stop; ++internal_it)
			*(first_out+internal_it) = transform_op( (first+internal_it), neighbour_op((first+internal_it)) );
	}, ex.concurrency_degree());


}


/**
\brief Invoke \ref md_stencil on multiple data sequences with
FF parallel execution.
\tparam InputIt Iterator type used for the input sequence.
\tparam OutputIt Iterator type used for the output sequence
\tparam Neighbourhood Callable type for obtaining the neighbourhood.
\tparam StencilTransformer Callable type for performing the stencil transformation.
\tparam OtherInputIts Iterator types for additional input sequences.
\param ex FF parallel execution policy object.
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param out Iterator to the first element in the output sequence.
\param transform_op Stencil transformation operation.
\param neighbour_op Neighbourhood operation.
\param other_firsts Iterators to the first element of additional input sequences.
*/
template <typename InputIt, typename OutputIt, typename StencilTransformer,
          typename Neighbourhood, typename ... OtherInputIts>
void stencil(parallel_execution_ff & ex,
             InputIt first, InputIt last, OutputIt first_out,
             StencilTransformer transform_op, Neighbourhood neighbour_op,
             OtherInputIts ... other_firsts ) {

	ssize_t total_parfor_size = last-first;
	ff::ParallelFor pf;

	pf.parallel_for_idx(0, total_parfor_size, 1, total_parfor_size/ex.concurrency_degree(),
			[&](const long internal_start, const long internal_stop, const int thid) {
		for (size_t internal_it = internal_start; internal_it < internal_stop; ++internal_it)
			*(first_out+internal_it) = transform_op( (first+internal_it),
					neighbour_op( (first+internal_it), (other_firsts+internal_it)... )
					);
	}, ex.concurrency_degree());

}


} // namespace
#endif

#endif /* GRPPI_FF_STENCIL_H_ */
