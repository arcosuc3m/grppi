/*
 * stream_reduce.h
 *
 *  Created on: Sep 4, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_STREAM_REDUCE_H
#define GRPPI_FF_STREAM_REDUCE_H

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"

#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/allocator.hpp>

#include "reduce.h"

// workers' actual task
template<typename T>
struct reduce_task_t {
	reduce_task_t(std::vector<T> &v) {
		vals.reserve(v.size());
		std::copy(v.begin(), v.end(), std::back_inserter(vals));
	}

	std::vector<T> vals;
};

namespace grppi {

// wraps the Generator function and sends
// the right number of items (according to window_size and offset) to workers
template<typename TSout, typename L>
struct ReduceEmitter : public ff::ff_node {
	using generated_type = typename std::result_of<L()>::type;
	using generated_value_type = typename generated_type::value_type;

    ReduceEmitter(int off, int wsz, L const & lf) :
    	winsize(wsz), offset(off), gen_func(lf) { }

    void * svc(void *) {
    	std::vector<generated_value_type> values;
    	values.reserve(winsize);

    	auto item = gen_func();
    	for(;;) {
    		while (item && values.size() != winsize) {
    			values.push_back(*item);
    			item = gen_func();
    		}
    		if (values.size()>0)
    			ff_send_out( new reduce_task_t<generated_value_type>(values) );

    		if (item) {
    			if (offset <= winsize)
    				values.erase(values.begin(), values.begin() + offset);
    			else {
    				values.erase(values.begin(), values.end());
    				auto diff = offset - winsize;
    				while (diff > 0 && item) {
    					item = gen_func();
    					diff--;
    				}
    			}
    		}
    		if (!item) break;
    	}
    	return EOS;
    }

    int winsize;
    int offset;
    L gen_func;
};

// wraps combiner function
template <typename TSin, typename TSout, typename L>
struct ReduceWorker : ff::ff_node {
	L combiner;
	TSout vaR_id;

	ReduceWorker(L const &comb, TSout ident) :
		combiner(comb), vaR_id(ident) { };

	void *svc(void *t) {
		reduce_task_t<TSin> *task = (reduce_task_t<TSin> *) t;
		grppi::sequential_execution seq{};

		void *outslot = ff::FFAllocator::instance()->malloc(sizeof(TSout));
		TSout *out = new (outslot) TSout();

		*out = grppi::reduce(seq, task->vals.begin(), task->vals.end(), vaR_id, std::forward<L>(combiner));

		task->vals.clear(); task->vals.shrink_to_fit();
		delete task;

		return (outslot);
	}
};

// wraps consumer function
template<typename TSin, typename L>
struct ReduceCollector : ff::ff_node {

	L cons_func;
	ReduceCollector(L const &lf) : cons_func(lf) {}

	void *svc(void *t) {
		TSin *task = (TSin *) t;
		cons_func(*task);

		ff::FFAllocator::instance()->free(task);
		return GO_ON;
	}
};



/**
\addtogroup stream_reduce_pattern
@{
\addtogroup stream_reduce_pattern_ff FastFlow parallel stream reduce pattern
\brief FF parallel implementation of the \ref md_stream-reduce.
@{
*/

/**
\brief Invoke \ref md_stream-reduce on a stream with
FF parallel execution.
\tparam Identity Type of the identity value used by the combiner.
\tparam Generator Callable type used for generating data items.
\tparam Combiner Callable type used for data items combination.
\tparam Consumer Callable type used for consuming data items.
\param ex FF parallel execution policy object.
\param window_size Number of consecutive items to be reduced.
\param offset Number of items after of which a new reduction is started.
\param identity Identity value for the combination.
\param generate_op Generation operation.
\param combine_op Combination operation.
\param consume_op Consume operation.
*/
template <typename Identity, typename Generator, typename Combiner,
          typename Consumer>

void stream_reduce(parallel_execution_ff & ex,
                   int window_size, int offset, Identity identity,
                   Generator &&generate_op,
				   Combiner &&combine_op,
                   Consumer &&consume_op) {

	bool notnested = true;

	using generator_type = typename std::result_of<Generator()>::type;
	using genOutType	 = typename generator_type::value_type;

	// According to the interface documentation, the Combiner can operate on different data types
	// [es: T res = cmb(Tx, U y)]. It is unclear the role of the identity here.

	auto nw = ex.concurrency_degree();

	// first stage
	std::unique_ptr<ff::ff_node> E = std::make_unique<ReduceEmitter<genOutType,Generator>>(offset, window_size, generate_op);

	// last stage
	std::unique_ptr<ff::ff_node> C = std::make_unique<ReduceCollector<Identity,Consumer>>(consume_op);

	std::vector<std::unique_ptr<ff::ff_node>> w;
	for(int i=0; i<nw; ++i)
		w.push_back( std::make_unique<ReduceWorker<genOutType,Identity,Combiner>>(combine_op, identity) );

	ff::ff_Farm<> farm( std::move(w), std::move(E), std::move(C) );

	farm.setFixedSize(true);
	farm.setInputQueueLength(nw*1);
	farm.setOutputQueueLength(nw*1);

	if(notnested)
		farm.run_and_wait_end();

}



} // namespace grppi

#endif


#endif /* GRPPI_FF_STREAM_REDUCE_H */
