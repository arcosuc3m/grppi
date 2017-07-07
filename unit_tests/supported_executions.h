#ifndef GRPPI_UNIT_TESTS_SUPPORTED_EXECUTION_H
#define GRPPI_UNIT_TESTS_SUPPORTED_EXECUTION_H

#include <gtest/gtest.h>

#include "common/sequential_execution.h"
#include "common/parallel_execution_native.h"
#include "common/parallel_execution_omp.h"
#include "common/parallel_execution_tbb.h"

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

#endif
