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
#ifndef GRPPI_DIVIDEANDCONQUER_H
#define GRPPI_DIVIDEANDCONQUER_H

#include "common/common.h"
#include "seq/divideandconquer.h"
#include "native/divideandconquer.h"
#include "omp/divideandconquer.h"
#include "tbb/divideandconquer.h"
#include "poly/divideandconquer.h"

#if 0 /* START DOCUMENTATION */
/** @addtogroup BDataPattern
 *  @{
 */
/** @defgroup DivideAndConquer
 *
 *	@brief Apply the divide and conquer pattern for parallelizing the code 
 *
 *  The DivideAndConquer divide a big problem in simpler problems and then
 *  solve each simpler problem in parallel using the 'task' function.
 *  Finally join the solution of all small problems using the 'merge' function.
 *
 *  @{
 */
/**
 *  @param exec     Execution_model flag to indicates the type of execution
 *    (sequential or parallel) and the implementation framework.
 *  @param problem  Data that we want to process.
 *  @param output   Memory address where to store the result of the operation.
 *  @param divide   DivideFunction: show how to divide the problem into simple.
 *    problems, until the data is simple enough to be handle by the 'task'
 *    function.
 *  @param task     Task function: will be executed once the problem is simple 
 *    enough and contains the code section that should be parallelize.
 *  @param merge    Merge function: will join the the result of the 'task' 
 *    function
 */
template <typename Input, typename Output, typename DivFunc, typename Operation, typename MergeFunc>
 void DivideAndConquer(execution_model exec, Input & problem, Output & output, DivFunc && divide, Operation && task, MergeFunc && merge);
/** @} */
/** @} */
#endif /* END DOCUMENTATION */

#endif
