#ifndef GRPPI_UNIT_TESTS_SUPPORTED_EXECUTION_H
#define GRPPI_UNIT_TESTS_SUPPORTED_EXECUTION_H

#include <gtest/gtest.h>

#include "seq/sequential_execution.h"
#include "native/parallel_execution_native.h"
#include "omp/parallel_execution_omp.h"
#include "tbb/parallel_execution_tbb.h"
#include "ff/parallel_execution_ff.h"

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

#ifdef GRPPI_FF
  ,
  grppi::parallel_execution_ff
#endif
>;

using executions_notbb = ::testing::Types<

  grppi::parallel_execution_native

#ifdef GRPPI_OMP
  ,
  grppi::parallel_execution_omp
#endif

>;

using executions_noff = ::testing::Types<

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
