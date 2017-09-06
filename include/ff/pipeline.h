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
#include <ff/allocator.hpp>
#include "ff_node_wrap.hpp"

namespace grppi {

// internals
class ff_wrap_pipe: public ff::ff_pipeline {
private:

	// base case
    template <typename ToutOfPrevStage>
        void add2pipeall() {}

    // Last stage
    template <typename ToutOfPrevStage, class Func> // Seq
    typename std::is_same<typename std::result_of<Func(ToutOfPrevStage)>::type,void>
        add2pipeall(Func const &stage) {
        auto n = new ff::PMINode<ToutOfPrevStage,void,FuncIn>(stage);
        cleanup_stages.push_back(n);
        ff::ff_pipeline::add_stage(n);
    }

    // Middle stages
    template <typename ToutOfPrevStage, class Func, typename ... Arguments> // Par
    typename std::is_same<decltype(Func::farmtype),int> // works by SFINAE -> fail if no farmtype field exists
    add2pipeall(Func const &stage, Arguments const &...args){
        //std::cout << "Parallel stage\n";
        auto taskf = *(stage.task);
        // Questo perché tipato è buono - fra l'altro al momento funziona solo con profondità 1
        // Si assume che Func sia un callable object sequenziale
        typedef typename std::remove_pointer<decltype(FuncIn::task)>::type task_t;
        typedef typename std::result_of<task_t(ToutOfPrevStage)>::type outItemType;
        typedef decltype(taskf) workerType;

        // Build the farm
        size_t nworkers = stage.exectype->num_threads;
        std::vector<std::unique_ptr<ff::ff_node>> w;
        for(int i=0; i<nworkers; ++i) w.push_back(std::make_unique<ff::PMINode<ToutOfPrevStage,outItemType,workerType> >(taskf));
        ff::ff_Farm<> * theFarm = new ff::ff_Farm<>(std::move(w)); //,std::move(E),std::move(C));
        theFarm->setFixedSize(true);
        theFarm->setInputQueueLength(nworkers*1);
        theFarm->setOutputQueueLength(nworkers*1);

        // Add farm to the pipe
        cleanup_stages.push_back(theFarm);
        ff::ff_pipeline::add_stage(theFarm);
        add2pipeall<outItemType, Arguments...>(args...);
    }

    // StreamFilter
    template <typename ToutOfPrevStage, class FuncIn, typename ... Arguments> // Par
    typename std::is_same<decltype(FuncIn::filtertype),int> // works by SFINAE -> fail if no filtertype field exists
    add2pipeall(FuncIn const &stage, Arguments const &...args) {
    	auto taskf = *(stage.task);
    	typedef decltype(taskf) workerType;
    	auto n = new ff::PMINodeFilter<ToutOfPrevStage,workerType>(taskf);
    	cleanup_stages.push_back(n);
    	ff::ff_pipeline::add_stage(n);
    	add2pipeall<ToutOfPrevStage, Arguments...>(args...);
    }


    template <typename ToutOfPrevStage, class FuncIn, typename ... Arguments>
    typename std::is_same<typename std::result_of<FuncIn(ToutOfPrevStage)>::type,void>
     add2pipeall(FuncIn const &stage, Arguments const &...args) {
        //if (is_callable<FuncIn(ToutOfPrevStage)>())
        typedef typename std::result_of<FuncIn(ToutOfPrevStage)>::type outItemType;
        //std::cout << "Sequential stage\n";

        auto n = new ff::PMINode<ToutOfPrevStage,outItemType,FuncIn>(stage);
        cleanup_stages.push_back(n);
        ff::ff_pipeline::add_stage(n);
        add2pipeall<outItemType, Arguments...>(args...);
    }


protected:
    std::vector<ff_node*> cleanup_stages;

public:
    template<typename First, typename... Rest>
    ff_wrap_pipe(First& stage, Rest&...stages) {
        typedef typename std::result_of<First()>::type outItemType; //::value_type outtype;
        outItemType ret;
        typedef decltype(ret.elem) deitemizedOutType;

        // First stage
        //auto n = new ff::PMINode<void,outItemType,First>(stage);
        auto n = new ff::PMINode<void,deitemizedOutType,First>(stage);
        cleanup_stages.push_back(n);
        ff::ff_pipeline::add_stage(n);

        // Other stages
        //this->add2pipeall<typename outItemType::value_type,Rest...>(stages...);
        this->add2pipeall<deitemizedOutType,Rest...>(stages...);
    }

    ~ff_wrap_pipe() {
        for (auto s: cleanup_stages) delete s;
    }

    operator ff_node* () { return this;}

    /*
    bool load_result(OUT_t *&task,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=ff_node::TICKS2WAIT) {
        return ff_pipeline::load_result((void**)&task, retry,ticks);
    }
    */

    // deleted members
    bool load_result(void ** task,
                     unsigned long retry=((unsigned long)-1),
                     unsigned long ticks=ff_node::TICKS2WAIT) = delete;


};

// TODO
template <typename Generator, typename ... Transformers,
          requires_no_arguments<Generator> = 0>
void pipeline(parallel_execution_ff & ex, Generator generate_op,
              Transformers && ... transform_ops) {

    bool notnested = true;

    ff_wrap_pipe pipe(in, sts ...);
    pipe.setFixedSize(true);
    pipe.setXNodeInputQueueLength(1024);
    pipe.setXNodeOutputQueueLength(1024);
    if (notnested) {
        pipe.run_and_wait_end();
    }

}

} // namespace

#endif // GRPPI_FF


#endif /* INCLUDE_FF_PIPELINE_H_ */
