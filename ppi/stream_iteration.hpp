/**
* @version		GrPPI v0.1
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

#ifndef PPI_STREAM_ITERATION
#define PPI_STREAM_ITERATION

#include "common/common.hpp"

#include "ppi_seq/stream_iteration_seq.hpp"
#include "ppi_thr/stream_iteration_thr.hpp"
//#include "ppi_thrust/farm_thrust.hpp"

#ifdef OMP_ENABLE
//	#include "ppi_omp/farm_omp.hpp"
#endif

#ifdef TBB_ENABLE
//	#include "ppi_tbb/farm_tbb.hpp"
#endif
#if 0 /* START DOCUMENTATION */
/** @addtogroup BStreamPattern
 *  @{
 */
/** @defgroup Farm
 *
 *  @brief Apply the farm pattern for parallelizing the code section
 *
 *  The Farm pattern apply a function 'taskf' to every independent element 
 *  returned by the generator function 'in'. The 'in' function read a data
 *	stream and forwards the result to the 'taskf' function. The task function
 *	is executed in parallel for as many thread as the user indicates in the
 *	'exec' variable.
 *  @{
 */
/** @param exec Execution_model flag to indicates the type of execution
 *    (sequential or parallel) and the implementation framework
 *  @param in   Generator function: This function determine how to read the data
 *    before start the parallel stage
 *  @param taskf Task function: Function that contains the code section that 
 *    should be parallelize
 */
template <typename GenFunc, typename TaskFunc>
void Farm(execution_model exec, GenFunc const &in, TaskFunc const & taskf);
/** @} */
/** @} */
#endif /* END DOCUMENTATION */

#endif
