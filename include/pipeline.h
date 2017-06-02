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

#ifndef GRPPI_PIPELINE_H
#define GRPPI_PIPELINE_H

#include "include/common/common.h"

#include "include/ppi_seq/pipeline_seq.h"
#include "include/ppi_thr/pipeline_thr.h"

#ifdef GRPPI_OMP
	#include "include/ppi_omp/pipeline_omp.h"
#endif

#ifdef GRPPI_THRUST
  #include "ppi_thrust/pipeline_thrust.hpp"
#endif 

#ifdef GRPPI_TBB
	#include "include/ppi_tbb/pipeline_tbb.h"
#endif 

#if 0 /* START DOCUMENTATION */
/** @addtogroup BStreamPattern
 *  @{
 */
/** @defgroup Pipeline
 *	@brief Apply the pipeline pattern for parallelizing the code section.
 *
 *	The Pipeline pattern is applied to data that is processed during 
 *	several steps.
 *  In a Pipeline each step is delegate to a different thread, each thread
 *  perform the function related with that step and forward the result to 
 *  the next step.
 *  @{
 */

/** @param exec Execution_model flag to indicates the type of execution
 *    (sequential or parallel) and the implementation framework.
 *  @param in   Generator function: This function determine how to read the data
 *    before start the parallel stage.
 *  @param sts  Task functions: one or more task functions sorted by the order
 *    of execution. Each task function will be executed in a different thread.
 */
template <typename FuncIn, typename ... Arguments>
void Pipeline(execution_model exec, FuncIn in, Arguments ... sts ) {
/** @} */
/** @} */
#endif /* END DOCUMENTATION */

#endif
