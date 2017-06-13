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

#ifndef GRPPI_THRUST_POLICY_H
#define GRPPI_THRUST_POLICY_H

// Only if compiled with Thrust enabled
#ifdef GRPPI_THRUST

#include <thrust/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>

// Only if additionally TBB is enabled
// Note: This allows TBB/Thrust integration.
#ifdef GRPPI_TBB
#include <thrust/system/tbb/execution_policy.h>
#endif

namespace grppi{

template< typename Policy >
class parallel_execution_thrust_internal {
  bool ordering = true;
  int num_threads;
public:
   bool lockfree = false;
   int num_gpus = 1;
   Policy policy;
   parallel_execution_thrust_internal(int _gpus, Policy _policy) : num_gpus{_gpus}, policy{_policy} {};
};

template<typename Policy>
parallel_execution_thrust_internal <Policy> parallel_execution_thrust ( int _gpus, Policy policy){
   return parallel_execution_thrust_internal<Policy>(_gpus, policy);
}

parallel_execution_thrust_internal<thrust::system::cuda::detail::par_t > parallel_execution_thrust(){
   return parallel_execution_thrust_internal<thrust::system::cuda::detail::par_t>(1, thrust::cuda::par);
}

parallel_execution_thrust_internal<thrust::system::cuda::detail::par_t > parallel_execution_thrust(int _gpus){
   return parallel_execution_thrust_internal<thrust::system::cuda::detail::par_t>(_gpus, thrust::cuda::par);
}

/*
auto parallel_execution_thrust() -> decltype(parallel_execution_thrust_internal(1, thrust::cuda::par))
{
    return parallel_execution_thrust_internal(1, thrust::cuda::par);
} 
*/

#endif // GRPPI_THRUST

#endif
