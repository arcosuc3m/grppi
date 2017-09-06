/*
 * divideconquer.h
 *
 *  Created on: Sep 6, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_DIVIDECONQUER_H
#define GRPPI_FF_DIVIDECONQUER_H

#ifdef GRPPI_FF

#include <ff/node.hpp>
#include <ff/dc.hpp>
#include "ff_node_wrap.hpp"

#include "parallel_execution_ff.h"


namespace grppi {

/**
\addtogroup divide_conquer_pattern
@{
\addtogroup divide_conquer_pattern_ff FastFlow parallel divide/conquer pattern.
\brief FF parallel implementation of the \ref md_divide-conquer.
@{
*/

/**
\brief Invoke \ref md_divide-conquer with FF
parallel execution.
\tparam Input Type used for the input problem.
\tparam Divider Callable type for the divider operation.
\tparam Solver Callable type for the solver operation.
\tparam Combiner Callable type for the combiner operation.
\param ex Sequential execution policy object.
\param input Input problem to be solved.
\param divider_op Divider operation.
\param solver_op Solver operation.
\param combiner_op Combiner operation.
*/
template <typename Input, typename Divider, typename Solver, typename Combiner>
typename std::result_of<Solver(Input)>::type
divide_conquer(parallel_execution_ff & ex, Input & input,
                   Divider && divide_op,
				   Solver && solve_op,
                   Combiner && combine_op) {

	using Output = typename std::result_of<Solver(Input)>::type;
	auto nw = ex.concurrency_degree();

	// divide
	std::function<void (const Input&, std::vector<Input> &)> divide_fn = [&](const Input & in, std::vector<Input> & subin) {
		subin = std::move(divide_op(in));
	};

	// conquer
	auto combine_fn = [&] (std::vector<Output>& in, Output& out) {
		for(auto i=0; i<in.size(); ++i) {
			combine_op(in[i], out);
		}
	};

	// sequential solver (base-case)
	auto seq_fn = [&] (const Input & in , Output & out) {
		out = solve_op(in);
	};

	// condition
	// TODO: grPPI interface does not consider splitting operand.
	// split condition is supposed to be included in the divide function.
	auto cond_fn = [&] (const Input &in) {
		if(in>2) return false;
		else return true;
		//return false;
	};

	Output out;
	// divide, combine, seq, condition, operand, res, wrks
	ff::ff_DC<Input,Output> dac(divide_fn,combine_fn,seq_fn,cond_fn,input,out,nw);
	dac.run_and_wait_end();

	return out;


}




} // namespace grppi

#endif



#endif /* INCLUDE_FF_DIVIDECONQUER_H_ */
