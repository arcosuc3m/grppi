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

#ifndef GRPPI_STREAM_ITERATION_H
#define GRPPI_STREAM_ITERATION_H

#include "common/sequential_execution.h"
#include "common/parallel_execution_native.h"

#include "seq/stream_iteration.h"
#include "native/stream_iteration.h"

#if 0 /* START DOCUMENTATION */
/** @addtogroup BStreamPattern
 *  @{
 */
/** @defgroup Farm
 *
 *  @brief Apply the farm pattern for parallelizing the code section
 *
 *  The Farm pattern apply a function 'op' to every independent element 
 *  returned by the generator function 'in'. The 'in' function read a data
 *	stream and forwards the result to the 'op' function. The task function
 *	is executed in parallel for as many thread as the user indicates in the
 *	'exec' variable.
 *  @{
 */
/** @param exec Execution_model flag to indicates the type of execution
 *    (sequential or parallel) and the implementation framework
 *  @param in   Generator function: This function determine how to read the data
 *    before start the parallel stage
 *  @param op Task function: Function that contains the code section that 
 *    should be parallelize
 */
template <typename GenFunc, typename Operation>
void Farm(execution_model exec, GenFunc &&in, Operation && op);
/** @} */
/** @} */
#endif /* END DOCUMENTATION */

#endif
