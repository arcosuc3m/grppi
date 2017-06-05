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

#ifndef GRPPI_STREAMFILTER_H
#define GRPPI_STREAMFILTER_H

#include "common/common.h"

#include "ppi_seq/stream_filter_seq.h"
#include "ppi_thr/stream_filter_thr.h"

#ifdef GRPPI_OMP
	#include "ppi_omp/stream_filter_omp.h"
#endif

#ifdef GRPPI_TBB
	#include "ppi_tbb/stream_filter_tbb.h"
#endif 

#if 0 /* START DOCUMENTATION */
/** @addtogroup BStreamPattern
 *  @{
 */
/** @defgroup StreamFilter
 *
 *  @brief Apply the stream filter pattern for parallelizing the code section
 *
 *  The Stream Filter pattern apply a boolean function 'filter' to every element
 *  of a data stream. If the function returns true for a given stream element
 *  then the 'out' function, that shows how to handle the output,
 *
 *  The Stream Filter has 3 steps:@n
 *    1) Read inputs from stream, as indicated with function 'in' (1 thread)@n
 *    2) Filter inputs, it is performed in parallel (N-1 threads)@n
 *    3) Handle output with function 'out' (1 thread)@n
 *  where N is the number of thread indicated by the user in 'exec'@n
 *  @{
 */
/** @param exec   Execution_model flag to indicates the type of execution
 *    (sequential or parallel) and the implementation framework.
 *  @param in     Generator function: This function determine how to read 
 *    the data before start the parallel stage.
 *  @param filter Filter function: Boolean function that contains the code 
 *    filter section. This code is going to be parallelize.
 *  @param out    Output Function: This function determine how to handle
 *    the filter output
 */
template <typename GenFunc, typename FilterFunc, typename OutFunc>
void StreamFilter(execution_model exec, GenFunc const & in, FilterFunc const & filter, OutFunc const & out );
/** @} */
/** @} */
#endif /* END DOCUMENTATION */

#endif
