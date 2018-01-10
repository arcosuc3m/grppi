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

#ifndef GRPPI_FF_PARALLEL_EXECUTION_FF_H
#define GRPPI_FF_PARALLEL_EXECUTION_FF_H

#ifdef GRPPI_FF

#include "../common/iterator.h"
#include "../common/execution_traits.h"
#include "../common/patterns.h"
#include "../common/farm_pattern.h"

#include <type_traits>
#include <tuple>

#include "internals/ff_streaming_wraps.hpp"

#include <ff/parallel_for.hpp>
#include <ff/dc.hpp>


namespace grppi {

/**
\brief FastFlow (FF) parallel execution policy.

This policy uses FastFlow as implementation back-end.
 */
class parallel_execution_ff {
public:

	/**
	  \brief Default construct a FF parallel execution policy.

	  Creates a FF parallel execution object.

	  The concurrency degree is determined by the platform.

	 */
	parallel_execution_ff() noexcept :
	parallel_execution_ff{get_physical_cores()}
	{}

	/**
	  \brief Constructs a FF parallel execution policy.

	  Creates a FF parallel execution object selecting the concurrency degree.

	  \param concurrency_degree Number of threads used for parallel algorithms.
	  \param order Whether ordered executions is enabled or disabled.
	 */
	parallel_execution_ff(int concurrency_degree, bool order=true) noexcept :
			concurrency_degree_{concurrency_degree},
			ordering_{order}
	{}

	/**
	  \brief Set number of grppi threads.
	 */
	void set_concurrency_degree(int degree) noexcept { concurrency_degree_ = degree; }

	/**
	  \brief Get number of grppi threads.
	 */
	int concurrency_degree() const noexcept { return concurrency_degree_; }

	/**
	  \brief Enable ordering.
	 */
	void enable_ordering() noexcept { ordering_= true; }

	/**
	  \brief Disable ordering.
	 */
	void disable_ordering() noexcept { ordering_= false; }

	/**
	  \brief Is execution ordered.
	 */
	bool is_ordered() const noexcept { return ordering_; }

	/**
	  \brief Applies a transformation to multiple sequences leaving the result in
	  another sequence using available FF parallelism.
	  \tparam InputIterators Iterator types for input sequences.
	  \tparam OutputIterator Iterator type for the output sequence.
	  \tparam Transformer Callable object type for the transformation.
	  \param firsts Tuple of iterators to input sequences.
	  \param first_out Iterator to the output sequence.
	  \param sequence_size Size of the input sequences.
	  \param transform_op Transformation callable object.
	  \pre For every I iterators in the range
	       `[get<I>(firsts), next(get<I>(firsts),sequence_size))` are valid.
	  \pre Iterators in the range `[first_out, next(first_out,sequence_size)]` are valid.
	 */
	template <typename ... InputIterators, typename OutputIterator, typename Transformer>
	void map(std::tuple<InputIterators...> firsts,
			OutputIterator first_out,
			std::size_t sequence_size,
			Transformer transform_op) const;

	/**
	  \brief Applies a reduction to a sequence of data items.
	  \tparam InputIterator Iterator type for the input sequence.
	  \tparam Identity Type for the identity value.
	  \tparam Combiner Callable object type for the combination.
	  \param first Iterator to the first element of the sequence.
	  \param sequence_size Size of the input sequence.
	  \param last Iterator to one past the end of the sequence.
	  \param identity Identity value for the reduction.
	  \param combine_op Combination callable object.
	  \pre Iterators in the range `[first,last)` are valid.
	  \return The reduction result.
	 */
	template <typename InputIterator, typename Identity, typename Combiner>
	auto reduce(InputIterator first,
			std::size_t sequence_size,
			Identity && identity,
			Combiner && combine_op) const;

	/**
	  \brief Applies a map/reduce operation to a sequence of data items.
	  \tparam InputIterator Iterator type for the input sequence.
	  \tparam Identity Type for the identity value.
	  \tparam Transformer Callable object type for the transformation.
	  \tparam Combiner Callable object type for the combination.
	  \param first Iterator to the first element of the sequence.
	  \param sequence_size Size of the input sequence.
	  \param identity Identity value for the reduction.
	  \param transform_op Transformation callable object.
	  \param combine_op Combination callable object.
	  \pre Iterators in the range `[first,last)` are valid.
	  \return The map/reduce result.
	 */
	template <typename ... InputIterators, typename Identity,
	typename Transformer, typename Combiner>
	auto map_reduce(std::tuple<InputIterators...> firsts,
			std::size_t sequence_size,
			Identity && identity,
			Transformer && transform_op,
			Combiner && combine_op) const;

	/**
	  \brief Applies a transformation to multiple sequences leaving the result in
	  another sequence.
	  \tparam InputIterators Iterator types for input sequences.
	  \tparam OutputIterator Iterator type for the output sequence.
	  \tparam Transformer Callable object type for the transformation.
	  \param firsts Tuple of iterators to input sequences.
	  \param first_out Iterator to the output sequence.
	  \param sequence_size Size of the input sequences.
	  \param transform_op Transformation callable object.
	  \pre For every I iterators in the range
	       `[get<I>(firsts), next(get<I>(firsts),sequence_size))` are valid.
	  \pre Iterators in the range `[first_out, next(first_out,sequence_size)]` are valid.
	 */
	template <typename ... InputIterators, typename OutputIterator,
	typename StencilTransformer, typename Neighbourhood>
	void stencil(std::tuple<InputIterators...> firsts,
			OutputIterator first_out,
			std::size_t sequence_size,
			StencilTransformer && transform_op,
			Neighbourhood && neighbour_op) const;

	/**
	  \brief Invoke \ref md_divide-conquer.
	  \tparam Input Type used for the input problem.
	  \tparam Divider Callable type for the divider operation.
	  \tparam Solver Callable type for the solver operation.
	  \tparam Combiner Callable type for the combiner operation.
	  \param input Input problem to be solved.
	  \param divider_op Divider operation.
	  \param solver_op Solver operation.
	  \param combine_op Combiner operation.
	 */
	template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
	auto divide_conquer(Input & input,
			Divider && divide_op,
			Predicate && condition_op,
			Solver && solve_op,
			Combiner && combine_op) const;

	/**
	  \brief Invoke \ref md_pipeline.
	  \tparam Generator Callable type for the generator operation.
	  \tparam Transformers Callable types for the transformers in the pipeline.
	  \param generate_op Generator operation.
	  \param transform_ops Transformer operations.
	 */
	template <typename Generator, typename ... Transformers>
	void pipeline(Generator && generate_op,
			Transformers && ... transform_op) const;


private:
  constexpr static int default_concurrency_degree = 4;
  int concurrency_degree_ = get_physical_cores();
  bool ordering_ = true; // needed in order to comply with unit_tests

};


// --------------------- METAFUNCTIONS ---------------------

/**
\brief Metafunction that determines if type E is parallel_execution_ff
\tparam Execution policy type.
 */
template <typename E>
constexpr bool is_parallel_execution_ff() {
	return std::is_same<E, parallel_execution_ff>::value;
}

/**
\brief Metafunction that determines if type E is supported in the current build.
\tparam Execution policy type.
 */
template <typename E>
constexpr bool is_supported();

/**
\brief Specialization stating that parallel_execution_ff is supported.
This metafunction evaluates to false if GRPPI_FF is enabled.
 */
template <>
constexpr bool is_supported<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_map<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_reduce<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_map_reduce<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_stencil<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the divideANDconquer pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_divide_conquer<parallel_execution_ff>() { return true; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for parallel_execution_ff when GRPPI_FF is enabled.
*/
template <>
constexpr bool supports_pipeline<parallel_execution_ff>() { return true; }


// -------------------- PUBLIC FUNCTIONS IMPLEMENTATION --------------------

// divide&conquer pattern
template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_ff::divide_conquer(Input & input,
		Divider && divide_op,
		Predicate && condition_op,
		Solver && solve_op,
		Combiner && combine_op) const {

	using output_type = typename std::result_of<Solver(Input)>::type;

	// divide
	auto divide_fn = [&](const Input &in, std::vector<Input> &subin) {
		subin = divide_op(in);
	};

	// combine
	auto combine_fn = [&] (std::vector<output_type>& in, output_type& out) {
			out = combine_op(in[0], in[1]);
	};

	// sequential solver (base-case)
	auto seq_fn = [&] (const Input & in , output_type & out) {
		out = solve_op(in);
	};

	// condition
	auto cond_fn = [&] (const Input &in) {
		return condition_op(in);
	};

	output_type out_var;
	// divide, combine, seq, condition, operand, res, wrks
	ff::ff_DC<Input,output_type> dac(divide_fn, combine_fn, seq_fn, cond_fn, input, out_var, concurrency_degree_);

	dac.run_and_wait_end();

	return out_var;
}

// map pattern
template <typename ... InputIterators, typename OutputIterator, typename Transformer>
void parallel_execution_ff::map(std::tuple<InputIterators...> firsts,
		OutputIterator first_out,
		std::size_t sequence_size,
		Transformer transform_op) const {
	ff::ParallelFor pf(concurrency_degree_, true);

	pf.parallel_for(0, sequence_size,
			[&](const long internal_it) {
			*(first_out+internal_it) = apply_iterators_indexed(transform_op, firsts, internal_it);
	}, concurrency_degree_);
}

// reduce pattern
template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_ff::reduce(InputIterator first,
		std::size_t sequence_size,
		Identity && identity,
		Combiner && combine_op) const {
	ff::ParallelForReduce<Identity> pfr(concurrency_degree_, true);

	Identity vaR = identity;
	Identity vaR_identity = identity;

	auto final_red = [&](typename std::iterator_traits<InputIterator>::value_type a,
			typename std::iterator_traits<InputIterator>::value_type r) {
		vaR = combine_op(a, r);
	};

	pfr.parallel_reduce(vaR, vaR_identity, 0, sequence_size,
			[&](const long internal_it, typename std::iterator_traits<InputIterator>::value_type &vaR) {
		vaR = combine_op( vaR, *(first+internal_it) );
	}, final_red, concurrency_degree_);

	return vaR;
}

// map+reduce pattern
template <typename ... InputIterators, typename Identity,
          typename Transformer, typename Combiner>
auto parallel_execution_ff::map_reduce(std::tuple<InputIterators...> firsts,
    std::size_t sequence_size,
    Identity && identity,
    Transformer && transform_op,
	Combiner && combine_op) const {

	ff::ParallelForReduce<Identity> pfr(concurrency_degree_, true);

	Identity vaR = identity;
	Identity var_id = identity;
	std::vector<Identity> partial_outs(sequence_size);

	// Map function
	auto Map = [&](const long internal_it) {
			partial_outs[internal_it] = apply_iterators_indexed(transform_op, firsts, internal_it);
	};

	// Reduce function - this is the partial reduce function, executed in parallel
	auto Reduce = [&](const long internal_it, Identity &vaR) {
		vaR = combine_op( vaR, partial_outs[internal_it] );
	};

	// Final reduce - this reduces partial results and is executed sequentially
	auto final_red = [&](Identity a, Identity b) {
		vaR = combine_op(a, b);
	};

	pfr.parallel_for(0, sequence_size, Map, concurrency_degree_ );
	pfr.parallel_reduce(vaR, var_id, 0, sequence_size, Reduce, final_red, concurrency_degree_);

	return vaR;
}

// stencil pattern
template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
void parallel_execution_ff::stencil(std::tuple<InputIterators...> firsts,
		OutputIterator first_out,
		std::size_t sequence_size,
		StencilTransformer && transform_op,
		Neighbourhood && neighbour_op) const {

	ff::ParallelFor pf(concurrency_degree_, true);

	pf.parallel_for(0, sequence_size,
			[&](const long internal_it) {
		const auto first_it = std::get<0>(firsts);
		auto next_chunks = iterators_next(firsts, internal_it);
		*(first_out+internal_it) = transform_op( (first_it+internal_it),
					apply_increment(neighbour_op, next_chunks) );
	}, concurrency_degree_);
}


// pipeline pattern
template <typename Generator, typename ... Transformers>
void parallel_execution_ff::pipeline(
    Generator && generate_op,
    Transformers && ... transform_ops) const {
	ff_wrap_pipeline pipe(concurrency_degree_, ordering_, generate_op, std::forward<Transformers>(transform_ops)...);

	pipe.setFixedSize(false);

	pipe.run_and_wait_end();
}


// -------------------- PRIVATE FUNCTIONS IMPLEMENTATION --------------------



} // end namespace grppi

#else // GRPPI_FF undefined

namespace grppi {


/// Parallel execution policy.
/// Empty type if GRPPI_FF disabled.
struct parallel_execution_ff {};

/**
\brief Metafunction that determines if type E is parallel_execution_ff
This metafunction evaluates to false if GRPPI_FF is disabled.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_ff() {
  return false;
}

}

#endif // GRPPI_FF


#endif /* GRPPI_FF_PARALLEL_EXECUTION_FF_H */

