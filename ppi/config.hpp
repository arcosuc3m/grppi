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

#ifndef PPI_CONFIGFILE
#define PPI_CONFIGFILE



#include "enable_flags.hpp"

#include <thread>
#include "common/mpmc_queue.hpp"


#ifdef __CUDACC__
	#include <thrust/execution_policy.h>
	#include <thrust/system/omp/execution_policy.h>

	#ifdef TBB_ENABLE
		#include <thrust/system/tbb/execution_policy.h>
	#endif
#endif

constexpr int BOOST_QUEUE_SIZE=1024;
constexpr size_t TBB_NTOKENS = 24;

#include "common/common.hpp"




#endif
