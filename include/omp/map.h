/*
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

#ifndef GRPPI_OMP_MAP_H
#define GRPPI_OMP_MAP_H

#ifdef GRPPI_OMP

#include "parallel_execution_omp.h"

namespace grppi {

template <typename InputIt, typename OutputIt, typename Transformer,
          typename ... OtherInputIts>
void internal_map(parallel_execution_omp & ex, 
                  InputIt first, InputIt last, 
                  OutputIt first_out,
                  Transformer && transf_op, 
                  int i, 
                  int elemperthr, OtherInputIts ... more_firsts)
{
  //Calculate local input and output iterator 
  auto begin = first + (elemperthr * i);
  auto end = first + (elemperthr * (i+1));
  if( i == ex.concurrency_degree()-1) end = last;
  auto out = first_out + (elemperthr * i);
  advance_iterators(elemperthr*i, more_firsts ...);
  while(begin!=end){
    *out = transf_op(*begin, *more_firsts ...);
    advance_iterators(more_firsts ...);
    begin++;
    out++;
  }
}

/**
\addtogroup map_pattern
@{
\addtogroup map_pattern_omp OpenMP parallel map pattern.
\brief OpenMP parallel implementation of the \ref md_map.
@{
*/

/**
\brief Invoke \ref md_map on a data sequence with OpenMP
parallel execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\param ex Parallel OpenMP execution policy object
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param transf_op Transformation operation.
*/
template <typename InputIt, typename OutputIt, typename Transformer>
void map(parallel_execution_omp & ex, 
         InputIt first, InputIt last, 
         OutputIt first_out, 
         Transformer && transf_op)
{
  int numElements = last - first;

  int elemperthr = numElements/ex.concurrency_degree();

  #pragma omp parallel
  {
   #pragma omp single nowait
   {
    for(int i=1;i<ex.concurrency_degree();i++){
      


      #pragma omp task firstprivate(i)
      {
        auto begin = first + (elemperthr * i);
        auto end = first + (elemperthr * (i+1));
        if(i == ex.concurrency_degree() -1 ) end = last;
        auto out = first_out + (elemperthr * i);
        while(begin!=end){
          *out = transf_op(*begin);
          begin++;
          out++;
        }
      }
     }
      //Map main threads
      auto beg =first;
      auto out = first_out;
      auto end = first+elemperthr;
      while(beg!=end){
            *out = transf_op(*beg);
            beg++;
            out++;
      }
      #pragma omp taskwait
    }
  }
}

/**
\brief Invoke \ref md_map on a data sequence with OpenMP
execution.
\tparam InputIt Iterator type used for input sequence.
\tparam OtuputIt Iterator type used for the output sequence.
\tparam Transformer Callable type for the transformation operation.
\tparam OtherInputIts Iterator types used for additional input sequences.
\param ex Parallel OpenMP execution policy object
\param first Iterator to the first element in the input sequence.
\param last Iterator to one past the end of the input sequence.
\param first_out Iterator to first elemento of the output sequence.
\param op Transformation operation.
\param more_firsts Additional iterators with first elements of additional sequences.
*/
template <typename InputIt, typename OutputIt, typename Transformer,
          typename ... OtherInputIts>
void map(parallel_execution_omp & ex, 
         InputIt first, InputIt last, 
         OutputIt first_out, 
         Transformer && transf_op, 
         OtherInputIts ... more_firsts)
{
  //Calculate number of elements per thread
  int numElements = last - first;
  int elemperthr = numElements/ex.concurrency_degree();

  //Create tasks
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      for(int i=1;i<ex.concurrency_degree();i++){

      #pragma omp task firstprivate(i)
      {
        internal_map(ex, first, last, first_out,
                     std::forward<Transformer>(transf_op) , i, elemperthr, 
                     more_firsts ...);
      }
      //End task
     }

     //Map main thread
     internal_map(ex, first,last, first_out, 
                  std::forward<Transformer>(transf_op), 0, elemperthr, 
                  more_firsts ...);

    //Join threads
    #pragma omp taskwait
    }
  }
}

/**
@}
@}
*/
}

#endif

#endif
