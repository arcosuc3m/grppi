/**
* @version		GrPPI v0.3
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
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

#ifndef GRPPI_FF_DETAIL_PIPELINE_IMPL_H
#define GRPPI_FF_DETAIL_PIPELINE_IMPL_H

#ifdef GRPPI_FF

#include "simple_node.h"
#include "ordered_stream_reduce.h"
#include "unordered_stream_reduce.h"
#include "ordered_stream_filter.h"
#include "unordered_stream_filter.h"
#include "iteration_nodes.h"
#include "../../common/mpmc_queue.h"


#include <ff/allocator.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

#include <atomic>

namespace grppi {

namespace detail_ff {


class pipeline_impl : public ff::ff_pipeline {
public:

  template <typename Generator, typename ... Transformers>
  pipeline_impl(int nworkers, bool ordered, Generator && gen, 
      Transformers && ... transform_ops);

private:

   /**
  \brief Sets the attributes for the queues built through make_queue<T>()
  */
  void set_queue_attributes(int size, queue_mode mode) noexcept {
    queue_size_ = size;
    queue_mode_ = mode;
  }

  /**
  \brief Makes a communication queue for elements of type T.
  Constructs a queue using the attributes that can be set via 
  set_queue_attributes(). The value is returned via move semantics.
  \tparam T Element type for the queue.
  */
  template <typename T>
  mpmc_queue<T> make_queue() const {
    return {queue_size_, queue_mode_};
  }

  void add_node(std::unique_ptr<ff_node> && p_node) {
    ff::ff_pipeline::add_stage(p_node.get());
    nodes_.push_back(std::forward<std::unique_ptr<ff_node>>(p_node));
  }

  template <typename Input, typename Transformer,
      requires_no_pattern<Transformer> = 0>
  auto add_stages(Transformer &&stage) 
  {
    using input_type = std::decay_t<Input>;
    using node_type = node_impl<input_type,void,Transformer>;

    auto p_stage = std::make_unique<node_type>(std::forward<Transformer>(stage));
    add_node(std::move(p_stage));
  }

  template <typename Input, typename Transformer, typename ... OtherTransformers,
      requires_no_pattern<Transformer> = 0>
  auto add_stages(Transformer && transform_op,
      OtherTransformers && ... other_transform_ops) 
  {
    static_assert(!std::is_void<Input>::value,
        "Transformer must take non-void argument");
    using output_type =
        std::decay_t<typename std::result_of<Transformer(Input)>::type>;
    static_assert(!std::is_void<output_type>::value,
        "Transformer must return a non-void result");

    using node_type = node_impl<Input,output_type,Transformer>;
    auto p_stage = std::make_unique<node_type>(
        std::forward<Transformer>(transform_op));

    add_node(std::move(p_stage));
    add_stages<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Input, typename ... Transformers,
          template <typename...> class Pipeline,
          typename ... OtherTransformers,
          requires_pipeline<Pipeline<Transformers...>> = 0>
  auto add_stages(Pipeline<Transformers...> & pipeline_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    return this->template add_stages<Input>(std::move(pipeline_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Input, typename ... Transformers,
          template <typename...> class Pipeline,
          typename ... OtherTransformers,
          requires_pipeline<Pipeline<Transformers...>> = 0>
  auto add_stages(Pipeline<Transformers...> && pipeline_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    return this->template add_stages_nested<Input>(
        std::tuple_cat(
          pipeline_obj.transformers(),
          std::forward_as_tuple(other_transform_ops...)
        ),
        std::make_index_sequence<sizeof...(Transformers)+sizeof...(OtherTransformers)>());
  }

  template <typename Input, typename ... Transformers, std::size_t ... I>
  auto add_stages_nested(std::tuple<Transformers...> && transform_ops,
      std::index_sequence<I...>) 
  {
    return add_stages<Input>(std::forward<Transformers>(std::get<I>(transform_ops))...);
  }

  template <typename Input, typename FarmTransformer,
          template <typename> class Farm,
          requires_farm<Farm<FarmTransformer>> = 0>
  auto add_stages(Farm<FarmTransformer> & farm_obj) 
  {
    return this->template add_stages<Input>(std::move(farm_obj));
  }

  template <typename Input, typename FarmTransformer,
          template <typename> class Farm,
          requires_farm<Farm<FarmTransformer>> = 0>
  auto add_stages(Farm<FarmTransformer> && farm_obj) 
  {
    static_assert(!std::is_void<Input>::value,
        "Farm must take non-void argument");
    using output_type = std::decay_t<typename std::result_of<
        FarmTransformer(Input)>::type>;

    using worker_type = node_impl<Input,output_type,Farm<FarmTransformer>>;
    std::vector<std::unique_ptr<ff::ff_node>> workers;
    for(int i=0; i<nworkers_; ++i) {
      workers.push_back(std::make_unique<worker_type>(
          std::forward<Farm<FarmTransformer>>(farm_obj))
      );
    }

    if(ordered_) {
      using node_type = ff::ff_OFarm<Input,output_type>;
      auto p_farm = std::make_unique<node_type>(std::move(workers));
      add_node(std::move(p_farm));
    } 
    else {
      using node_type = ff::ff_Farm<Input,output_type>;
      auto p_farm = std::make_unique<node_type>(std::move(workers));
      add_node(std::move(p_farm));
    }
  }

  // parallel stage -- Farm pattern ref with variadic
  template <typename Input, typename FarmTransformer,
          template <typename> class Farm,
          typename ... OtherTransformers,
          requires_farm<Farm<FarmTransformer>> = 0>
  auto add_stages(Farm<FarmTransformer> & farm_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    return this->template add_stages<Input>(std::move(farm_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  // parallel stage -- Farm pattern with variadic
  template <typename Input, typename FarmTransformer,
          template <typename> class Farm,
          typename ... OtherTransformers,
          requires_farm<Farm<FarmTransformer>> = 0>
  auto add_stages( Farm<FarmTransformer> && farm_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    static_assert(!std::is_void<Input>::value,
        "Farm must take non-void argument");
    using output_type =
        std::decay_t<typename std::result_of<FarmTransformer(Input)>::type>;
    static_assert(!std::is_void<output_type>::value,
        "Farm must return a non-void result");

    using worker_type = node_impl<Input,output_type,Farm<FarmTransformer>>;
    std::vector<std::unique_ptr<ff::ff_node>> workers;

    for(int i=0; i<nworkers_; ++i) {
      workers.push_back(std::make_unique<worker_type>(
          std::forward<Farm<FarmTransformer>>(farm_obj))
      );
    }

    if(ordered_) {
      using node_type = ff::ff_OFarm<Input,output_type>;
      auto p_farm = std::make_unique<node_type>(std::move(workers));
      add_node(std::move(p_farm));
      add_stages<output_type>(std::forward<OtherTransformers>(other_transform_ops)...);
    } 
    else {
      using node_type = ff::ff_Farm<Input,output_type>;
      auto p_farm = std::make_unique<node_type>(std::move(workers));
      add_node(std::move(p_farm));
      add_stages<output_type>(std::forward<OtherTransformers>(other_transform_ops)...);
    }
  }

  // parallel stage -- Filter pattern ref
  template <typename Input, typename Predicate,
      template <typename> class Filter,
      requires_filter<Filter<Predicate>> = 0>
  auto add_stages(Filter<Predicate> & filter_obj) 
  {
    return this->template add_stages<Input>(std::move(filter_obj));
  }

  // parallel stage -- Filter pattern
  template <typename Input, typename Predicate,
      template <typename> class Filter,
      requires_filter<Filter<Predicate>> = 0>
  auto add_stages(Filter<Predicate> && filter_obj) 
{
    static_assert(!std::is_void<Input>::value,
        "Filter must take non-void argument");

    if(ordered_) {
      using node_type = ordered_stream_filter<Input,Filter<Predicate>>;
      auto p_farm = std::make_unique<node_type>(
          std::forward<Filter<Predicate>>(filter_obj), nworkers_);
      add_node(std::move(p_farm));
    } 
    else {
      using node_type = unordered_stream_filter<Input,Filter<Predicate>>;
      auto p_farm = std::make_unique<node_type>(
          std::forward<Filter<Predicate>>(filter_obj), nworkers_);
      add_node(std::move(p_farm));
    }
  }

  // parallel stage -- Filter pattern ref with variadics
  template <typename Input, typename Predicate,
      template <typename> class Filter,
      typename ... OtherTransformers,
      requires_filter<Filter<Predicate>> = 0>
  auto add_stages(Filter<Predicate> & filter_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    return this->template add_stages<Input>(std::move(filter_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  // parallel stage -- Filter pattern with variadics
  template <typename Input, typename Predicate,
      template <typename> class Filter,
      typename ... OtherTransformers,
      requires_filter<Filter<Predicate>> = 0>
  auto add_stages(Filter<Predicate> && filter_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    static_assert(!std::is_void<Input>::value,
        "Filter must take non-void argument");

    if(ordered_) {
      using node_type = ordered_stream_filter<Input,Filter<Predicate>>;
      auto p_farm = std::make_unique<node_type>(
              std::forward<Filter<Predicate>>(filter_obj), nworkers_);
      add_node(std::move(p_farm));
      add_stages<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
    } 
    else {
      using node_type = unordered_stream_filter<Input,Filter<Predicate>>;
      auto p_farm = std::make_unique<node_type>(
          std::forward<Filter<Predicate>>(filter_obj), nworkers_);
      add_node(std::move(p_farm));
      add_stages<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
    }
  }

  template <typename Input, typename Combiner, typename Identity,
          template <typename C, typename I> class Reduce,
          typename ... OtherTransformers,
          requires_reduce<Reduce<Combiner,Identity>> = 0>
  auto add_stages(Reduce<Combiner,Identity> & reduce_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    return this->template add_stages<Input>(std::move(reduce_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }

  template <typename Input, typename Combiner, typename Identity,
          template <typename C, typename I> class Reduce,
          typename ... OtherTransformers,
          requires_reduce<Reduce<Combiner,Identity>> = 0>
  auto add_stages(Reduce<Combiner,Identity> && reduce_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    static_assert(!std::is_void<Input>::value,
        "Reduce must take non-void argument");

    if(ordered_) {
      using reducer_type = Reduce<Combiner,Identity>;
      using node_type = ordered_stream_reduce<Input,reducer_type,Combiner>;
      auto p_farm = std::make_unique<node_type>(
          std::forward<reducer_type>(reduce_obj), 
          nworkers_);
      add_node(std::move(p_farm));
      add_stages<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
    } 
    else {
      using reducer_type = Reduce<Combiner,Identity>;
      using node_type = unordered_stream_reduce<Input,reducer_type,Combiner>;
      auto p_farm = std::make_unique<node_type>(
          std::forward<Reduce<Combiner,Identity>>(reduce_obj), 
          nworkers_);
      add_node(std::move(p_farm));
      add_stages<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
    }
  }

  /**
  \brief Adds a stage with an iteration object.
  \note This version takes iteration by l-value reference.
  */
  template <typename Input, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_no_pattern<Transformer> =0>
  auto add_stages(Iteration<Transformer,Predicate> & iteration_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    return this->template add_stages<Input>(std::move(iteration_obj),
        std::forward<OtherTransformers>(other_transform_ops)...);
  }


  /**
  \brief Adds a stage with an iteration object.
  \note This version takes iteration by r-value reference.
  */
  template <typename Input, typename Transformer, typename Predicate,
            template <typename T, typename P> class Iteration,
            typename ... OtherTransformers,
            requires_iteration<Iteration<Transformer,Predicate>> =0,
            requires_no_pattern<Transformer> =0>
  auto add_stages(Iteration<Transformer,Predicate> && iteration_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    std::vector<std::unique_ptr<ff::ff_node>> workers;

    using iteration_type = Iteration<Transformer,Predicate>;
    using worker_type = iteration_worker<Input,iteration_type>;
    for (int i=0; i<nworkers_; ++i)
      workers.push_back(
        std::make_unique<worker_type>(
          std::forward<iteration_type>(iteration_obj)));

    if (ordered_) {
      using node_type = ff::ff_OFarm<Input>;
      auto p_farm = std::make_unique<node_type>(std::move(workers));
      add_node(std::move(p_farm));
      add_stages<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
    } 
    else {
      using node_type = ff::ff_Farm<Input>;
      auto p_farm = std::make_unique<node_type>(std::move(workers));
      add_node(std::move(p_farm));
      add_stages<Input>(std::forward<OtherTransformers>(other_transform_ops)...);
    }
  }

  /**
  \brief Adds a stage with an iteration object.
  \note This version takes iteration by r-value reference.
  \note This version takes as body of the iteration an inner pipeline.
  */
  template <typename Input, typename Transformer, typename Predicate,
      template <typename T, typename P> class Iteration,
      typename ... OtherTransformers,
      requires_iteration<Iteration<Transformer,Predicate>> =0,
      requires_pipeline<Transformer> =0>
  auto add_stages(Iteration<Transformer,Predicate> && iteration_obj,
      OtherTransformers && ... other_transform_ops) 
  {
    static_assert(!is_pipeline<Transformer>, "Not implemented");
  }

  template <typename Input, typename Execution, typename Transformer, 
            template <typename, typename> class Context,
            typename ... OtherTransformers,
            requires_context<Context<Execution,Transformer>> = 0>
  auto add_stages(Context<Execution,Transformer> & context_op, 
       OtherTransformers &&... other_ops)
  {
    return this->template add_stages<Input>(std::move(context_op),
      std::forward<OtherTransformers>(other_ops)...);
  }

  template <typename Input, typename Execution, typename Transformer, 
            template <typename, typename> class Context,
            typename ... OtherTransformers,
            requires_context<Context<Execution,Transformer>> = 0>
  auto add_stages(Context<Execution,Transformer> && context_op, 
       OtherTransformers &&... other_ops)
  {

   return this->template add_stages<Input>(context_op.transformer(),
      std::forward<OtherTransformers>(other_ops)...);
  }


private:

  int nworkers_;
  bool ordered_;
  std::vector<std::unique_ptr<ff_node>> nodes_;
  
  queue_mode queue_mode_ = queue_mode::blocking;

  constexpr static int default_queue_size = 100;
  int queue_size_ = default_queue_size;

};

template <typename Generator, typename ... Transformers>
pipeline_impl::pipeline_impl(
    int nworkers, 
    bool ordered, 
    Generator && gen_op, 
    Transformers && ... transform_ops)
  :
    nworkers_{nworkers}, 
    ordered_{ordered},
    nodes_{}
{
  using result_type = std::decay_t<typename std::result_of<Generator()>::type>;
  using generator_value_type = typename result_type::value_type;
  using node_type = node_impl<void,generator_value_type,Generator>;

  auto first_stage = std::make_unique<node_type>(
      std::forward<Generator>(gen_op));

  add_node(std::move(first_stage));

  add_stages<generator_value_type>(std::forward<Transformers>(transform_ops)...);
}


} // namespace detail_ff

} // namespace grppi

#else

#endif // GRPPI_FF

#endif
