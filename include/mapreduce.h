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

#ifndef GRPPI_MAPREDUCE_H 
#define GRPPI_MAPREDUCE_H

#include "common/common.h"
#include "seq/mapreduce.h"
#include "native/mapreduce.h"
#include "omp/mapreduce.h"
#include "tbb/mapreduce.h"
#include "poly/mapreduce.h"

#if 0 /* START DOCUMENTATION */
/** @addtogroup BDataPattern
 *  @{
 */
/** @defgroup Mapreduce
 *
 *  @brief Apply the mapreduce pattern for parallelizing the code section.
 *
 *	The Mapreduce pattern is based on the application of a function 'map' 
 *	to all the elements of a given data structures 'in', and then merge
 * 	the result to another variable using binary elemental operations.
 *	In a Mapreduce the datastructure is divided in as many data sections as 
 *	threads are available to be used by the function that will execute in
 *	in parallel the code section 'map'. Each task will reduce its own 
 *	data section. Finaly all the thread section will be join in an output 
 *	variable.
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
 *	@param map 			Map function: this function will be applied to all the
 *		elements in the data structure in the interval first/end.
 *	@param op				Reduction elementary operation: this is a elemental
 *		binary operator. std::plus, std::minuns, std::multiplies, std::divides...
 *	@param inputs		Extra inputs to be applied to the function 'map'
 */
template < typename InputIt, typename OutputIt, typename MapFunc, typename ReduceOperator, typename ... MoreIn >
void MapReduce (execution_model exec, InputIt first, InputIt last, OutputIt firstOut, MapFunc && map, ReduceOperator op, MoreIn ... inputs);
/** @} */
/** @} */
#endif /* END DOCUMENTATION */

#endif
