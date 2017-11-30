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

#ifndef GRPPI_STREAM_POOL_H
#define GRPPI_STREAM_POOL_H


namespace grppi {

/** 
\addtogroup stream_patterns
@{
\defgroup stream_pool_pattern Stream Pool pattern
\brief Interface for applyinng the \ref md_stream_pool.
@{
*/


/**
\brief Invoke \ref md_stream_pool.
\tparam Population Type for the initial population.
\tparam Selection Callable type for the selection operation.
\tparam Selection Callable type for the evolution operation.
\tparam Selection Callable type for the evaluation operation.
\tparam Selection Callable type for the termination operation.
\param population initial population.
\param selection_op Selection operation.
\param evolution_op Evolution operations.
\param eval_op Evaluation operation.
\param termination_op Termination operation.
*/
template <typename Execution, typename Population, typename Selection, 
            typename Evolution, typename Evaluation, typename Predicate>
void stream_pool(Execution & ex,
                Population & population,
                Selection && selection_op,
                Evolution && evolve_op,
                Evaluation && eval_op,
                Predicate && termination_op)
{
  ex.stream_pool(ex,population, std::forward<Selection>(selection_op),
       std::forward<Evolution>(evolve_op), std::forward<Evaluation>(eval_op),
       std::forward<Predicate>(termination_op));
}

/**
@}
@}
*/

}

#endif
