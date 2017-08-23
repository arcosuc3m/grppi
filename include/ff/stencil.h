/*
 * stencil.h
 *
 *  Created on: Aug 23, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_STENCIL_H
#define GRPPI_FF_STENCIL_H

#ifdef GRPPI_FF

#include "parallel_execution_ff.h"


namespace grppi {

// TODO:
template <typename InputIt, typename OutputIt, typename StencilTransformer,
          typename Neighbourhood>
void stencil(parallel_execution_tbb & ex,
             InputIt first, InputIt last, OutputIt first_out,
             StencilTransformer transform_op,
             Neighbourhood neighbour_op) {

}


// TODO:
template <typename InputIt, typename OutputIt, typename StencilTransformer,
          typename Neighbourhood, typename ... OtherInputIts>
void stencil(parallel_execution_tbb & ex,
             InputIt first, InputIt last, OutputIt first_out,
             StencilTransformer transform_op, Neighbourhood neighbour_op,
             OtherInputIts ... other_firsts ) {

}


} // namespace
#endif

#endif /* GRPPI_FF_STENCIL_H_ */
