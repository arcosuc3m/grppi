/*
 * ff_pipeline_wrap.hpp
 *
 *  Created on: 16 Oct 2017
 *      Author: fabio
 */

#ifndef FF_PIPELINE_WRAP_HPP
#define FF_PIPELINE_WRAP_HPP

#include "../common/iterator.h"
#include "../common/execution_traits.h"
#include "../common/patterns.h"
#include "../common/farm_pattern.h"

#include <type_traits>
#include <tuple>

#include "ff_node_wrap.hpp"

#include <ff/farm.hpp>
#include <ff/pipeline.hpp>


namespace grppi {

// wraps stream reduce worker
template<typename TSin, typename Reducer>
struct ReduceWorker : ff::ff_node_t<TSin> {

	ReduceWorker(const Reducer& red_t) :
		_reduction_obj(red_t) { }

	TSin *svc(TSin *t) {
		TSin *result = t;
		std::experimental::optional<TSin> check;
		constexpr grppi::sequential_execution seq;

		check = *t;
		if(check) {
			_reduction_obj.add_item(std::forward<TSin>(check.value()));
			if (_reduction_obj.reduction_needed())
				*result = _reduction_obj.reduce_window(seq);
		} else return {};

		return result;
	}
	// Class variables
	Reducer _reduction_obj;
};

// wrapper for stream iteration worker
template<typename TSin, typename Iterator>
struct IterationWorker : ff::ff_node_t<TSin> {

	IterationWorker(const Iterator& iter_t) :
		_iterator_obj(iter_t) { }

	TSin *svc(TSin *t) {
		TSin *item = (TSin*) t;
		std::experimental::optional<TSin> check;

		check = *t;
		if(check) {
			do {
				*item = _iterator_obj.transform(std::forward<TSin>(*item));
			} while (!_iterator_obj.predicate(std::forward<TSin>(*item)));

			return item;
		} else return {};
	}
	// class variable
	Iterator _iterator_obj;
};


// Class supporting FastFlow backend for pipeline pattern
// constructs a pipeline whose stages can be sequential stages or
// nested streaming patterns.
class ff_wrap_pipeline: public ff::ff_pipeline {
private:

	inline void add2pipe(ff_node &node) {
		ff::ff_pipeline::add_stage(&node);
	}
	inline void add2pipe(ff_node *node) {
		cleanup_stages.push_back(node);
		ff::ff_pipeline::add_stage(node);
	}

	// base case -- empty
	template <typename T>
	void add2pipeall() { }

	// ALL OTHER CASES

	// last stage -- no output value expected
	template <typename Input, typename Transformer,
	requires_no_pattern<Transformer> = 0>
	auto add2pipeall(Transformer &&stage) {
		auto n = new ff::PMINode<Input,void,Transformer>(std::forward<Transformer>(stage));
		add2pipe(n);
	}

	// sequential stage -- intermediate stage, no pattern
	template <typename Input, typename Transformer, typename ... OtherTransformers,
	requires_no_pattern<Transformer> = 0>
	auto add2pipeall(Transformer && transform_op,
			OtherTransformers && ... other_transform_ops) {
		static_assert(!std::is_void<Input>::value,
				"Transformer must take non-void argument");
		using output_type =
				std::decay_t<typename std::result_of<Transformer(Input)>::type>;
		static_assert(!std::is_void<output_type>::value,
				"Transformer must return a non-void result");

		auto n = new ff::PMINode<Input,output_type,Transformer>(std::forward<Transformer>(transform_op));

		// add node to pipeline
		add2pipe(n);

		// recurse - template deduction should stop this recursion when no more args are given
		add2pipeall<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
	}

	// parallel stage -- Farm pattern
	template <typename Input, typename FarmTransformer,
	template <typename> class Farm,
	requires_farm<Farm<FarmTransformer>> = 0>
	auto add2pipeall( Farm<FarmTransformer> && farm_obj) {
		static_assert(!std::is_void<Input>::value,
				"Farm must take non-void argument");
		using output_type =
				std::decay_t<typename std::result_of<FarmTransformer(Input)>::type>;

		std::vector<std::unique_ptr<ff::ff_node>> w;

		for(int i=0; i<nworkers; ++i)
			w.push_back( std::make_unique<ff::PMINode<Input,output_type,Farm<FarmTransformer>>>(
					std::forward<Farm<FarmTransformer>>(farm_obj))
					);

		ff::ff_OFarm<Input,output_type> * theFarm = new ff::ff_OFarm<Input,output_type>(std::move(w));

		// Add farm to the pipeline
		add2pipe(theFarm);
	}

	// parallel stage -- Farm pattern with variadic
	template <typename Input, typename FarmTransformer,
	template <typename> class Farm,
	typename ... OtherTransformers,
	requires_farm<Farm<FarmTransformer>> = 0>
	auto add2pipeall( Farm<FarmTransformer> && farm_obj,
			OtherTransformers && ... other_transform_ops) {
		static_assert(!std::is_void<Input>::value,
				"Farm must take non-void argument");
		using output_type =
				std::decay_t<typename std::result_of<FarmTransformer(Input)>::type>;
		static_assert(!std::is_void<output_type>::value,
				"Farm must return a non-void result");

		std::vector<std::unique_ptr<ff::ff_node>> w;

		for(int i=0; i<nworkers; ++i)
			w.push_back( std::make_unique<ff::PMINode<Input,output_type,Farm<FarmTransformer>>>(
					std::forward<Farm<FarmTransformer>>(farm_obj))
					);

		ff::ff_OFarm<Input,output_type> * theFarm = new ff::ff_OFarm<Input,output_type>(std::move(w));

		// Add farm to the pipeline
		add2pipe(theFarm);

		// recurse - template deduction stops the recursion when no more args are given
		add2pipeall<Input>( std::forward<OtherTransformers>(other_transform_ops)... );
	}

	// parallel stage -- Filter pattern
	template <typename Input, typename Predicate,
	template <typename> class Filter,
	requires_filter<Filter<Predicate>> = 0>
	auto add2pipeall(Filter<Predicate> && filter_obj) {
		static_assert(!std::is_void<Input>::value,
				"Filter must take non-void argument");

		std::vector<std::unique_ptr<ff::ff_node>> w;

		for(int i=0; i<nworkers; ++i)
			w.push_back( std::make_unique<ff::PMINodeFilter<Input,Filter<Predicate>>>(std::forward<Filter<Predicate>>(filter_obj)) );

		ff::ff_OFarm<Input> * theFarm = new ff::ff_OFarm<Input>(std::move(w));

		// add node to pipeline
		add2pipe(theFarm);
	}

	// parallel stage -- Filter pattern with variadics
	template <typename Input, typename Predicate,
	template <typename> class Filter,
	typename ... OtherTransformers,
	requires_filter<Filter<Predicate>> = 0>
	auto add2pipeall(Filter<Predicate> && filter_obj,
			OtherTransformers && ... other_transform_ops) {
		static_assert(!std::is_void<Input>::value,
				"Filter must take non-void argument");

		std::vector<std::unique_ptr<ff::ff_node>> w;

		for(int i=0; i<nworkers; ++i)
			w.push_back( std::make_unique<ff::PMINodeFilter<Input,Filter<Predicate>>>(std::forward<Filter<Predicate>>(filter_obj)) );

		ff::ff_OFarm<Input> * theFarm = new ff::ff_OFarm<Input>(std::move(w));

		// add node to pipeline
		add2pipe(theFarm);

		// recurse - template deduction stops the recursion when no more args are given
		add2pipeall<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
	}

	// parallel stage -- Reduce pattern with variadics
	template <typename Input, typename Combiner, typename Identity,
	template <typename C, typename I> class Reduce,
	typename ... OtherTransformers,
	requires_reduce<Reduce<Combiner,Identity>> = 0>
	auto add2pipeall(Reduce<Combiner,Identity> && reduce_obj,
			OtherTransformers && ... other_transform_ops) {
		static_assert(!std::is_void<Input>::value,
				"Reduce must take non-void argument");

		std::vector<std::unique_ptr<ff::ff_node>> w;

		for(int i=0; i<nworkers; ++i)
			w.push_back( std::make_unique<ReduceWorker<Input,Reduce<Combiner,Identity>>>(
					std::forward<Reduce<Combiner,Identity>>(reduce_obj))
					);

		ff::ff_OFarm<Input> * theFarm = new ff::ff_OFarm<Input>( std::move(w) );

		// add node to pipeline
		add2pipe(theFarm);

		// recurse - template deduction stops the recursion when no more args are given
		add2pipeall<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
	}

	// parallel stage -- iterator pattern
	template <typename Input, typename Transformer, typename Predicate,
	template <typename T, typename P> class Iteration,
	typename ... OtherTransformers,
	requires_iteration<Iteration<Transformer,Predicate>> =0,
	requires_no_pattern<Transformer> =0>
	auto add2pipeall(Iteration<Transformer,Predicate> && iteration_obj,
			OtherTransformers && ... other_transform_ops) {

		std::vector<std::unique_ptr<ff::ff_node>> w;

		for(int i=0; i<nworkers; ++i)
			w.push_back( std::make_unique<IterationWorker<Input,Iteration<Transformer,Predicate>>>(
					std::forward<Iteration<Transformer,Predicate>>(iteration_obj))
					);

		ff::ff_OFarm<Input> * theFarm = new ff::ff_OFarm<Input>( std::move(w) );

		// add node to pipeline
		add2pipe(theFarm);

		// recurse - template deduction stops the recursion when no more args are given
		add2pipeall<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
	}

	// parallel stage -- iterator pattern
	template <typename Input, typename Transformer, typename Predicate,
	template <typename T, typename P> class Iteration,
	typename ... OtherTransformers,
	requires_iteration<Iteration<Transformer,Predicate>> =0,
	requires_pipeline<Transformer> =0>
	auto add2pipeall(Iteration<Transformer,Predicate> && iteration_obj,
			OtherTransformers && ... other_transform_ops) {
		static_assert(!is_pipeline<Transformer>, "Not implemented");
	}

	// pipeline of pipelines
	template <typename Input, typename ... Transformers,
	template <typename...> class Pipeline,
	typename ... OtherTransformers,
	requires_pipeline<Pipeline<Transformers...>> = 0>
	auto add2pipeall(Pipeline<Transformers...> && pipeline_obj,
			OtherTransformers && ... other_transform_ops) {
		add2pipe_nested<Input>(std::tuple_cat(pipeline_obj.transformers(),
				std::forward_as_tuple(other_transform_ops...)),
				std::make_index_sequence<sizeof...(Transformers)+sizeof...(OtherTransformers)>());
	}

	// nested type recursion
	template <typename Input, typename ... Transformers,
	std::size_t ... I>
	auto add2pipe_nested(std::tuple<Transformers...> && transform_ops,
			std::index_sequence<I...>) {
		return add2pipeall<Input>(std::forward<Transformers>(std::get<I>(transform_ops))...);
	}


protected:
	std::vector<ff_node*> cleanup_stages;
	size_t nworkers;

public:
	template<typename Generator, typename... Transformers>
	ff_wrap_pipeline(size_t nw, Generator& gen_func,
			Transformers&&...stages_ops) : nworkers{nw} {

				using result_type = std::decay_t<typename std::result_of<Generator()>::type>;
				using generator_value_type = typename result_type::value_type;

				// First stage
				auto n = new ff::PMINode<void,generator_value_type,Generator>(std::forward<Generator>(gen_func));
				add2pipe(n);

				// Other stages
				add2pipeall<generator_value_type>( std::forward<Transformers>(stages_ops)... );
			}

			~ff_wrap_pipeline() {
				for (auto s: cleanup_stages) delete s;
			}

			operator ff_node* () { return this;}
};

} // namespace




#endif /* FF_PIPELINE_WRAP_HPP */
