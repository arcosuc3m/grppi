/*
 * stream_filter.h
 *
 *  Created on: Sep 4, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_STREAM_FILTER_H
#define GRPPI_FF_STREAM_FILTER_H

#ifdef GRPPI_FF

#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/allocator.hpp>
#include "ff_node_wrap.hpp"

#include "parallel_execution_ff.h"


namespace grppi {

/**
\addtogroup filter_pattern
@{
*/

/**
\addtogroup filter_pattern_ff FastFlow parallel filter pattern.
\brief FF parallel implementation of the \ref md_stream-filter.
@{
*/

/**
\brief Invoke \ref md_stream-filter on a data
sequence with sequential execution policy.
This function keeps in the stream only those items
that satisfy the predicate.
\tparam Generator Callable type for value generator.
\tparam Predicate Callable type for filter predicate.
\tparam Consumer Callable type for value consumer.
\param ex FF parallel execution policy object.
\param generate_op Generator callable object.
\param predicate_op Predicate callable object.
\param consume_op Consumer callable object.
*/

template <typename Generator, typename Predicate, typename Consumer>
void keep(parallel_execution_ff & ex, Generator generate_op,
          Predicate predicate_op, Consumer consume_op) {

	bool notnested = true;

	using generator_type   = typename std::result_of<Generator()>::type;
	using generalOutType   = typename generator_type::value_type;
	//using transformer_type = typename std::result_of<Predicate(emitterOutType)>::type;
	//using transformerOutType = typename transformer_type::value_type;

	auto nw = ex.concurrency_degree();

	// first stage
	std::unique_ptr<ff::ff_node> E = std::make_unique<ff::PMINode<void,generalOutType,Generator>>(generate_op);
	// last stage
	std::unique_ptr<ff::ff_node> C = std::make_unique<ff::PMINode<generalOutType,void,Consumer>>(consume_op);

	std::vector<std::unique_ptr<ff::ff_node>> w;
	for(int i=0; i<nw; ++i)
		w.push_back( std::make_unique<ff::PMINodeFilter<generalOutType,Predicate>>(predicate_op) );

	ff::ff_Farm<> farm( std::move(w), std::move(E), std::move(C) );

	farm.setFixedSize(true);
	farm.setInputQueueLength(nw*1);
	farm.setOutputQueueLength(nw*1);

	if(notnested)
		farm.run_and_wait_end();



}


/**
\brief Invoke \ref md_stream-filter pattern on a data
sequence with sequential execution policy.
This function discards from the stream those items
that satisfy the predicate.
\tparam Generator Callable type for value generator.
\tparam Predicate Callable type for filter predicate.
\tparam Consumer Callable type for value consumer.
\param ex FF parallel execution policy object.
\param generate_op Generator callable object.
\param predicate_op Predicate callable object.
\param consume_op Consumer callable object.
*/

template <typename Generator, typename Predicate, typename Consumer>
void discard(parallel_execution_ff & ex, Generator generate_op,
             Predicate predicate_op, Consumer consume_op) {
	keep(ex,
			std::forward<Generator>(generate_op),
			[&](auto val) { return !predicate_op(val); },
			std::forward<Consumer>(consume_op)
	);

}


} // namespace grppi

#endif

#endif /* GRPPI_FF_STREAM_FILTER_H */
