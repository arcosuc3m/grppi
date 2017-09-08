/*
 * pipeline.h
 *
 *  Created on: Aug 23, 2017
 *      Author: fabio
 */

#ifndef INCLUDE_FF_PIPELINE_H_
#define INCLUDE_FF_PIPELINE_H_

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"

#include <experimental/optional>

#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include "ff_node_wrap.hpp"

namespace grppi {

// internals
class ff_wrap_pipeline: public ff::ff_pipeline {
private:

	// need to see this before add2pipeall variadic template function
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

    // Last stage
    template <typename Transformer, class Input>
    //typename std::is_same<typename std::result_of<Transformer(Input)>::type,void> // last stage node must return void?
    void add2pipeall(Input, Transformer &&stage) {

        auto n = new ff::PMINode<Input,void,Transformer>(stage);
        add2pipe(n);
    }

    // Intermediate stage (aka sequential stage)
    template <typename Transformer, typename ... MoreTransformers, typename Input>
    void add2pipeall(Input, Transformer && predicate_op, MoreTransformers && ... more_transform_ops) {

    	using optional_input_type = std::experimental::optional<Input>;
    	using output_type = typename std::result_of<Transformer(Input)>::type;
    	//using predOutType = typename output_type::value_type;
    	using optional_output_type = std::experimental::optional<output_type>;

    	auto n = new ff::PMINode<Input,output_type,Transformer>(predicate_op);

    	// add node to pipeline
    	add2pipe(n);

    	// recurse - template deduction should stop this recursion when no more args are given
    	add2pipeall(output_type{}, std::forward<MoreTransformers>(more_transform_ops)...);
    }


    // TODO: not clear how the compiler would deduce this case
    // Predicate (aka sequential filter)
    template <typename Predicate, typename ... MoreTransformers, typename Input>
    void add2pipeall(Input,
    		filter_info<parallel_execution_ff, Predicate> && predicate_op,
			MoreTransformers && ... more_transform_ops) {

    	using optional_input_type = std::experimental::optional<Input>;
    	using output_type = typename std::result_of<Predicate(Input)>::type;
    	//using predOutType = typename output_type::value_type;
    	using optional_output_type = std::experimental::optional<output_type>;

    	auto n = new ff::PMINodeFilter<Input,Predicate>(predicate_op.task);

    	// add node to pipeline
    	add2pipe(n);

    	// recurse - template deduction should stop this recursion when no more args are given
    	add2pipeall(output_type{}, std::forward<MoreTransformers>(more_transform_ops)...);
    }

    // TODO: not tested. farm nesting into pipeline fails with any execution case.
    template <typename Transformer, typename ... MoreTransformers, typename Input>
    void add2pipeall(Input,
    		farm_info<parallel_execution_ff,Transformer>  &stage,
			MoreTransformers const &...args) {
    	using farm_type = farm_info<parallel_execution_ff,Transformer>;

    	// output_type -- typedef typename std::remove_pointer<decltype(Transformer::task)>::type task_t;
    	// output_value_type -- using outItemType = typename std::result_of<task_t(ToutOfPrevStage)>::type ;

    	using output_type = typename std::result_of<Transformer(Input)>::type;
    	//using output_value_type = output_type::value_type;

    	// TODO: these types are used to handle optional values.
    	using optional_input_type = std::experimental::optional<Input>;
    	using optional_output_type = std::experimental::optional<output_type>;

    	// Build the farm
    	size_t nworkers = stage.exectype.concurrency_degree(); //stage.exectype->num_threads;

    	std::vector<std::unique_ptr<ff::ff_node>> w;

    	for(int i=0; i<nworkers; ++i)
    		w.push_back(std::make_unique<ff::PMINode<Input,output_type,Transformer> >(stage.task));


    	ff::ff_Farm<> * theFarm = new ff::ff_Farm<>(std::move(w)); //,std::move(E),std::move(C));
    	theFarm->setFixedSize(true);
    	theFarm->setInputQueueLength(nworkers*1);
    	theFarm->setOutputQueueLength(nworkers*1);

    	// Add farm to the pipeline
    	add2pipe(theFarm);

    	// recurse - template deduction should stop this recursion when no more args are given
    	add2pipeall(output_type{}, std::forward<MoreTransformers>(args)...);
    }


protected:
    std::vector<ff_node*> cleanup_stages;

public:
    template<typename Generator, typename... Transformers>
    ff_wrap_pipeline(Generator& gen_func, Transformers&...stages_ops) {

        using result_type = typename std::result_of<Generator()>::type;
        using generator_value_type = typename result_type::value_type;
        using output_type = std::experimental::optional<generator_value_type>;

        // First stage
        auto n = new ff::PMINode<void,generator_value_type,Generator>(gen_func);
        add2pipe(n);

        // Other stages
        add2pipeall(generator_value_type{}, std::forward<Transformers>(stages_ops)...);
    }

    ~ff_wrap_pipeline() {
        for (auto s: cleanup_stages) delete s;
    }

    operator ff_node* () { return this;}

    // deleted members
    bool load_result(void ** task,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=ff_node::TICKS2WAIT) = delete;


};


/**
\addtogroup pipeline_pattern
@{
\addtogroup pipeline_pattern_ff FastFlow parallel pipeline pattern
\brief FF parallel implementation of the \ref md_pipeline.
@{
*/

/**
\brief Invoke \ref md_pipeline on a data stream
with FF parallel execution.
\tparam Generator Callable type for the stream generator.
\tparam Transformers Callable type for each transformation stage.
\param ex FF parallel execution policy object.
\param generate_op Generator operation.
\param trasnform_ops Transformation operations for each stage.
\remark Generator shall be a zero argument callable type.
*/
template <typename Generator, typename ... Transformers,
          requires_no_arguments<Generator> = 0>
void pipeline(parallel_execution_ff & ex, Generator generate_op,
              Transformers && ... transform_ops) {

    bool notnested = true;

    ff_wrap_pipeline pipe(generate_op, transform_ops ...);

    pipe.setFixedSize(true);
    pipe.setXNodeInputQueueLength(1024);
    pipe.setXNodeOutputQueueLength(1024);

    if (notnested)
        pipe.run_and_wait_end();

}

} // namespace

#endif // GRPPI_FF


#endif /* INCLUDE_FF_PIPELINE_H_ */
