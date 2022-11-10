/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GRPPI_UNIT_TESTS_SUPPORTED_EXECUTION_H
#define GRPPI_UNIT_TESTS_SUPPORTED_EXECUTION_H

#include <gtest/gtest.h>

#include "grppi/seq/sequential_execution.h"
#include "grppi/native/parallel_execution_native.h"
#include "grppi/omp/parallel_execution_omp.h"
#include "grppi/tbb/parallel_execution_tbb.h"

using executions = ::testing::Types<
  grppi::sequential_execution,

  grppi::parallel_execution_native

#ifdef GRPPI_OMP
  ,
  grppi::parallel_execution_omp
#endif

#ifdef GRPPI_TBB
  ,
  grppi::parallel_execution_tbb
#endif

>;

using executions_notbb = ::testing::Types<
  grppi::sequential_execution,

  grppi::parallel_execution_native

#ifdef GRPPI_OMP
  ,
  grppi::parallel_execution_omp
#endif

>;

using executions_minimal = ::testing::Types<
    grppi::sequential_execution,
    grppi::parallel_execution_native
>;


#endif
