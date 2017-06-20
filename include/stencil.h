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

#ifndef GRPPI_STENCIL_H 
#define GRPPI_STENCIL_H

#include "common/common.h"

#include "seq/stencil.h"

#include "native/stencil.h"

#include "omp/stencil.h"

#include "tbb/stencil.h"


#if 0 /* START DOCUMENTATION */
/** @addtogroup BDataPattern
 *  @{
 */
/** @defgroup Stencil
 *
 *  @brief Apply the stencil pattern for parallelizing the code section.
 *
 *  The Stencil pattern is applied in data structures when the computations
 *  need information about other elements of the data structures.
 *  The Stencil divide the data structure in as many parts as threads are 
 *  available to be used. Then each threads perform the computation needed for
 *  each neighbor of that element using the 'neighbor' function and perform 
 *  the code section 'op' with the data computed with the neighbors.
 *  @{
 */
/** @param exec     Execution_model flag to indicates the type of execution
 *    (sequential or parallel) and the implementation framework and number of
 *    threads.
 *  @param first    Iterator pointing to the first element of the input data
 *    structure. 
 *  @param last     Iterator pointing to the last element of the input data
 *    structure.
 *  @param firstOut Iterator pointing to the first elements of the output data
 *    structure.
 *  @param op    Task function: function that contains the code section that 
 *    will be parallelized.
 *  @param neighbor Support function: function that handle how to get the needed
 *    data from the neighbors. This functions serves as support for getting data
 *    to 'op'.
 */
template <typename InputIt, typename OutputIt, typename Operation, typename NFunc>
 void Stencil(execution_model exec, InputIt first, InputIt last, OutputIt firstOut, Operation && op, NFunc && neighbor );
/** @} */
/** @} */
#endif /* END DOCUMENTATION */

#endif
