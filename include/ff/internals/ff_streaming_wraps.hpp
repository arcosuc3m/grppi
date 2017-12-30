/**
* @version		GrPPI v0.3
* @copyright	Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef FF_STREAMING_WRAPS_HPP
#define FF_STREAMING_WRAPS_HPP

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include "../../common/iterator.h"
#include "../../common/execution_traits.h"
#include "../../common/patterns.h"
#include "../../common/farm_pattern.h"

#include <type_traits>
#include <tuple>

#include "ff_helper_classes.hpp"
#include "ff_node_wrap.hpp"

#include <ff/farm.hpp>
#include <ff/pipeline.hpp>


namespace grppi {

// Class supporting FastFlow backend for pipeline
// pattern. Constructs a pipeline whose stages can
// be sequential stages or nested streaming patterns.
class ff_wrap_pipeline: public ff::ff_pipeline {
public:
	template<typename Generator, typename... Transformers>
	ff_wrap_pipeline(size_t nw, Generator&& gen_func, Transformers&&...stages_ops)
		: nworkers{nw} {
		using result_type = std::decay_t<typename std::result_of<Generator()>::type>;
		using generator_value_type = typename result_type::value_type;

		// First stage
		auto n = new ff::FFNode<void,generator_value_type,Generator>(std::forward<Generator>(gen_func));
		add2pipe(n);

		// Other stages
		add2pipeall<generator_value_type>( std::forward<Transformers>(stages_ops)... );
	}

	~ff_wrap_pipeline() {
		for (auto s: cleanup_stages) delete s;
	}

	operator ff_node* () { return this; }

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

	// -- ALL OTHER CASES

	// last stage -- no output value expected
	template <typename Input, typename Transformer,
	requires_no_pattern<Transformer> = 0>
	auto add2pipeall(Transformer &&stage) {
		using input_type = std::decay_t<Input>;

		auto n = new ff::FFNode<input_type,void,Transformer>(std::forward<Transformer>(stage));
		add2pipe(n);
	}

	// intermediate sequential stage -- no pattern
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

		auto n = new ff::FFNode<Input,output_type,Transformer>(std::forward<Transformer>(transform_op));

		add2pipe(n);
		add2pipeall<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
	}

	// parallel stage -- Farm pattern ref
	template <typename Input, typename FarmTransformer,
	template <typename> class Farm,
	requires_farm<Farm<FarmTransformer>> = 0>
	auto add2pipeall(Farm<FarmTransformer> & farm_obj) {
		return this->template add2pipeall<Input>(std::move(farm_obj));
	}

	// parallel stage -- Farm pattern
	template <typename Input, typename FarmTransformer,
	template <typename> class Farm,
	requires_farm<Farm<FarmTransformer>> = 0>
	auto add2pipeall(Farm<FarmTransformer> && farm_obj) {
		static_assert(!std::is_void<Input>::value,
				"Farm must take non-void argument");
		using output_type =
				std::decay_t<typename std::result_of<FarmTransformer(Input)>::type>;

		std::vector<std::unique_ptr<ff::ff_node>> w;

		for(int i=0; i<nworkers; ++i)
			w.push_back( std::make_unique<ff::FFNode<Input,output_type,Farm<FarmTransformer>>>(
					std::forward<Farm<FarmTransformer>>(farm_obj))
					);

		ff::ff_OFarm<Input,output_type> * theFarm = new ff::ff_OFarm<Input,output_type>(std::move(w));

		add2pipe(theFarm);
	}

	// parallel stage -- Farm pattern ref with variadic
	template <typename Input, typename FarmTransformer,
	template <typename> class Farm,
	typename ... OtherTransformers,
	requires_farm<Farm<FarmTransformer>> = 0>
	auto add2pipeall(Farm<FarmTransformer> & farm_obj,
			OtherTransformers && ... other_transform_ops) {
		return this->template add2pipeall<Input>(std::move(farm_obj),
				std::forward<OtherTransformers>(other_transform_ops)...);
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
			w.push_back( std::make_unique<ff::FFNode<Input,output_type,Farm<FarmTransformer>>>(
					std::forward<Farm<FarmTransformer>>(farm_obj))
					);

		ff::ff_OFarm<Input,output_type> * theFarm = new ff::ff_OFarm<Input,output_type>(std::move(w));

		add2pipe(theFarm);
		add2pipeall<Input>( std::forward<OtherTransformers>(other_transform_ops)... );
	}

	// parallel stage -- Filter pattern ref
	template <typename Input, typename Predicate,
	template <typename> class Filter,
	requires_filter<Filter<Predicate>> = 0>
	auto add2pipeall(Filter<Predicate> & filter_obj) {
		return this->template add2pipeall<Input>(std::move(filter_obj));
	}

	// parallel stage -- Filter pattern
	template <typename Input, typename Predicate,
	template <typename> class Filter,
	requires_filter<Filter<Predicate>> = 0>
	auto add2pipeall(Filter<Predicate> && filter_obj) {
		static_assert(!std::is_void<Input>::value,
				"Filter must take non-void argument");

		ff::ff_StreamFilter_grPPI<Input,Filter<Predicate>> *theFarm =
				new ff::ff_StreamFilter_grPPI<Input,Filter<Predicate>>(
						std::forward<Filter<Predicate>>(filter_obj), nworkers
				);

		add2pipe(theFarm);
	}

	// parallel stage -- Filter pattern ref with variadics
	template <typename Input, typename Predicate,
	template <typename> class Filter,
	typename ... OtherTransformers,
	requires_filter<Filter<Predicate>> = 0>
	auto add2pipeall(Filter<Predicate> & filter_obj,
			OtherTransformers && ... other_transform_ops) {
		return this->template add2pipeall<Input>(std::move(filter_obj),
				std::forward<OtherTransformers>(other_transform_ops)...);

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

		ff::ff_StreamFilter_grPPI<Input,Filter<Predicate>> *theFarm =
				new ff::ff_StreamFilter_grPPI<Input,Filter<Predicate>>(
						std::forward<Filter<Predicate>>(filter_obj), nworkers
				);

		add2pipe(theFarm);
		add2pipeall<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
	}

	// parallel stage -- Reduce pattern ref with variadics
	template <typename Input, typename Combiner, typename Identity,
	template <typename C, typename I> class Reduce,
	typename ... OtherTransformers,
	requires_reduce<Reduce<Combiner,Identity>> = 0>
	auto add2pipeall(Reduce<Combiner,Identity> & reduce_obj,
			OtherTransformers && ... other_transform_ops) {
		return this->template add2pipeall<Input>(std::move(reduce_obj),
				std::forward<OtherTransformers>(other_transform_ops)...);
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

		ff::ff_StreamReduce_grPPI<Input,Reduce<Combiner,Identity>> *theFarm =
				new ff::ff_StreamReduce_grPPI<Input,Reduce<Combiner,Identity>>(
						std::forward<Reduce<Combiner,Identity>>(reduce_obj), nworkers
				);

		add2pipe(theFarm);
		add2pipeall<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
	}

	// parallel stage -- iterator pattern ref with variadics
	template <typename Input, typename Transformer, typename Predicate,
	template <typename T, typename P> class Iteration,
	typename ... OtherTransformers,
	requires_iteration<Iteration<Transformer,Predicate>> =0,
	requires_no_pattern<Transformer> =0>
	auto add2pipeall(Iteration<Transformer,Predicate> & iteration_obj,
			OtherTransformers && ... other_transform_ops) {
		return this->template add2pipeall<Input>(std::move(iteration_obj),
				std::forward<OtherTransformers>(other_transform_ops)...);
	}

	// parallel stage -- iterator pattern with variadics
	template <typename Input, typename Transformer, typename Predicate,
	template <typename T, typename P> class Iteration,
	typename ... OtherTransformers,
	requires_iteration<Iteration<Transformer,Predicate>> =0,
	requires_no_pattern<Transformer> =0>
	auto add2pipeall(Iteration<Transformer,Predicate> && iteration_obj,
			OtherTransformers && ... other_transform_ops) {

		std::vector<std::unique_ptr<ff::ff_node>> w;

		for(int i=0; i<nworkers; ++i)
			w.push_back( std::make_unique<ff::IterationWorker<Input,Iteration<Transformer,Predicate>>>(
					std::forward<Iteration<Transformer,Predicate>>(iteration_obj))
					);

		ff::ff_OFarm<Input> * theFarm = new ff::ff_OFarm<Input>( std::move(w) );

		add2pipe(theFarm);
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

	// pipeline of pipelines ref
	template <typename Input, typename ... Transformers,
	template <typename...> class Pipeline,
	typename ... OtherTransformers,
	requires_pipeline<Pipeline<Transformers...>> = 0>
	auto add2pipeall(Pipeline<Transformers...> & pipeline_obj,
			OtherTransformers && ... other_transform_ops) {
		return this->template add2pipeall<Input>(std::move(pipeline_obj),
					std::forward<OtherTransformers>(other_transform_ops)...);
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
};

} // namespace



namespace { // -- utility functions

inline void run_command(const char *cmd, std::string& result) {
	std::array<char, 128> buffer;
	std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);

	if (!pipe) throw std::runtime_error("popen() failed!");

	result.clear();
	while (!feof(pipe.get()))
		if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
			result += buffer.data();
}

inline int get_physical_cores() {
	int count=1, nc, ns;
	std::string res{};

#if defined(__linux__)
	char cores[] = "lscpu | grep 'Core(s)' | awk '{print $4}'";
	char sockets[] = "lscpu | grep 'Socket(s)' | awk '{print $2}'";

	run_command(cores, res);
	nc = std::atoi(res.c_str());

	run_command(sockets, res);
	ns = std::atoi(res.c_str());

	count = nc / ns;

#elif defined (__APPLE__)
	char cmd[] = "sysctl hw.physicalcpu | awk '{print $2}'";

	run_command(cmd, res);
	count = std::stoi(res);

#else
#pragma message ("Cannot determine physical cores number on this platform")
#endif

	return count;
}

} // Anonymous namespace


#endif /* FF_STREAMING_WRAPS_HPP */
