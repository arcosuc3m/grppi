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

#ifndef PPI_MAPREDUCE_THRUST
#define PPI_MAPREDUCE_THRUST

#ifdef __CUDACC__

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/transform_reduce.h>

using namespace std;
namespace grppi{
template < typename InputIt, typename OutputIt, typename MapFunc, typename RedFunc, typename Policy, typename ... MoreIn >
inline void map_reduce (parallel_execution_thrust_internal<Policy>  p, InputIt first, InputIt last, OutputIt firstOut, MapFunc const & map, RedFunc const & reduce, MoreIn ... inputs) {
       
       thrust::device_vector< typename InputIt::value_type > d_vec_in(first, last);  
       *firstOut = thrust::transform_reduce(thrust::cuda::par, d_vec_in.begin(), d_vec_in.end(), map, 0, reduce);       
}

template < typename InputIt, typename Output, typename MapFunc, typename RedFunc, typename Policy >
inline void map_reduce (parallel_execution_thrust_internal<Policy>  p, InputIt first, InputIt last,  MapFunc const & map, RedFunc const & reduce, Output & out) {
       thrust::device_vector< typename InputIt::value_type > d_vec_in(first, last);
       out = thrust::transform_reduce(thrust::cuda::par, d_vec_in.begin(), d_vec_in.end(), map, 0, reduce);
}
}
#endif

#endif
