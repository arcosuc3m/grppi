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

		bool notnested = ff::outer_ff_pattern;
		ff::outer_ff_pattern = false;
		typedef typename std::result_of<Generator()>::type::type emitterOutType;
		typedef typename std::result_of<Consumer(emitterOutType)>::type workerOutType;
		auto nw = ex.concurrency_degree();

		std::unique_ptr<ff::ff_node> E = std::make_unique<ff::PMINode<void,emitterOutType,Generator>>(in);

		std::vector<std::unique_ptr<ff::ff_node>> w;
		for(int i=0; i<nw; ++i)
			w.push_back( std::make_unique<ff::PMINode<emitterOutType,void,Consumer> >(taskf) );

		ff::ff_Farm<> farm( std::move(w), std::move(E) );

		farm.setFixedSize(true);
		farm.setInputQueueLength(nw*1);
		farm.setOutputQueueLength(nw*1);

		if(notnested) {
			farm.remove_collector(); // needed to avoid init errors!
			farm.run_and_wait_end();

			// check if ff_nodes need to be deleted
			// in case of nested pattern this don't work - to be fixed with unique_ptr
			//for(int i=0;i<p.num_threads;++i) delete w[i];
		}
}

// TODO:
template <typename Generator, typename Transformer, typename Consumer>
void farm(parallel_execution_ff & ex, Generator generate_op,
          Transformer transform_op , Consumer consume_op) {

}



} // namespace
#endif


#endif /* GRPPI_FF_FARM_H_ */
