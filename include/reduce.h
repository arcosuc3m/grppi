/**
* @version		GrPPI v0.2
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

#ifndef GRPPI_REDUCE_H 
#define GRPPI_REDUCE_H

#include "common/common.h"

#include "common/iterator_traits.h"

#include "seq/reduce.h"

#include "native/reduce.h"

#include "omp/reduce.h"

#include "tbb/reduce.h"


#if 0 /* START DOCUMENTATION */
/** @addtogroup BDataPattern
 *  @{
 */
/** @defgroup Reduce
 *
 *  @brief Apply the reduce pattern for parallelizing the code section.
 *
 *	The Reduce pattern is based on the application of elemental associative
 *  operation between the elements of a data structure.
 *
 *  In the Reduce the input data structure is divided in as many data sections
 *  as threads are available to use.
 *  Each thread apply the elemental associative operation in all the elements 
 *  of its data section.
 *  Finally the result of all the threads will be merged in in an output
 *  variable using the elemental associative operation 'op'.
 *
 *  @{
 */
/** @param exec 		Execution_model flag to indicates the type of execution
 *    (sequential or parallel) and the implementation framework and number of
 *		threads.
 *  @param first  	Iterator pointing to the first element of the input data
 *		structure. 
 *  @param last	  	Iterator pointing to the last element of the input data
 *		structure.
 *	@param firstOut	Iterator pointing to the first elements of the output data
 *		structure.
 *	@param op				Reduction elementary operation: this is a elemental
 *		binary operator. std::plus, std::minuns, std::multiplies, std::divides...
 */
template < typename InputIt, typename Output, typename ReduceOperator>
void Reduce(execution_model exec, InputIt first, InputIt last, Output & firstOut, ReduceOperator op);
/** @} */
/** @} */
#endif /* END DOCUMENTATION */

#endif
