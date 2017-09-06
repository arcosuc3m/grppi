/*
 * farm.h
 *
 *  Created on: Aug 23, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_FARM_H
#define GRPPI_FF_FARM_H

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"

#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/allocator.hpp>
#include "ff_node_wrap.hpp"

namespace grppi {

/**
\addtogroup farm_pattern
@{
\addtogroup farm_pattern_ff FasfFlow parallel farm pattern
\brief FF parallel implementation of the \ref md_farm.
@{
*/

/**
\brief Invoke \ref md_farm on a data stream with FF parallel
execution with a generator and a consumer.
\tparam Generator Callable type for the generation operation.
\tparam Consumer Callable type for the consume operation.
\param ex FF parallel execution policy object.
\param generate_op Generator operation.
\param consume_op Consumer operation.
*/
template <typename Generator, typename Consumer>
void farm(parallel_execution_ff & ex, Generator generate_op,
		Consumer consume_op) {

	// keep this flag for nesting patterns.
	bool notnested = true;

	auto nw = ex.concurrency_degree();

	// Deduces the return type
	using generator_type = typename std::result_of<Generator()>::type;
	using emitterOutType = typename generator_type::value_type;

	// Consumer takes values of the result type of the Transformer. Since this interface
	// seems to be designed for identity farm, the consumer should consume data generated by
	// the generator
	using consumerOutType = typename std::result_of<Consumer(emitterOutType)>::type;

	std::unique_ptr<ff::ff_node> E = std::make_unique<ff::PMINode<void,emitterOutType,Generator> >(generate_op);

	std::vector<std::unique_ptr<ff::ff_node>> w;
	for(int i=0; i<nw; ++i)
		w.push_back( std::make_unique<ff::PMINode<emitterOutType,consumerOutType,Consumer> >(consume_op) );

	ff::ff_Farm<> farm( std::move(w), std::move(E) );

	farm.setFixedSize(true);
	farm.setInputQueueLength(nw*1);
	farm.setOutputQueueLength(nw*1);

	if(notnested) {
		farm.remove_collector();
		farm.run_and_wait_end();
	}
}

template <typename Generator, typename Transformer, typename Consumer>
void farm(parallel_execution_ff & ex, Generator generate_op,
		Transformer transform_op , Consumer consume_op) {

	bool notnested = true;

	using generator_type   = typename std::result_of<Generator()>::type;
	using emitterOutType   = typename generator_type::value_type;
	using transformer_type = typename std::result_of<Transformer(emitterOutType)>::type;
	//using transformerOutType = typename transformer_type::value_type;

	auto nw = ex.concurrency_degree();

	// first stage
	std::unique_ptr<ff::ff_node> E = std::make_unique<ff::PMINode<void,emitterOutType,Generator>>(generate_op);
	// last stage
	std::unique_ptr<ff::ff_node> C = std::make_unique<ff::PMINode<transformer_type,void,Consumer>>(consume_op);

	std::vector<std::unique_ptr<ff::ff_node>> w;
	for(int i=0; i<nw; ++i)
		w.push_back( std::make_unique<ff::PMINode<emitterOutType,transformer_type,Transformer>>(transform_op) );

	ff::ff_Farm<> farm( std::move(w), std::move(E), std::move(C) );

	farm.setFixedSize(true);
	farm.setInputQueueLength(nw*1);
	farm.setOutputQueueLength(nw*1);

	if(notnested)
		farm.run_and_wait_end();
}



} // namespace
#endif


#endif /* GRPPI_FF_FARM_H_ */
