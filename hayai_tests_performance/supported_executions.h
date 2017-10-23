#ifndef GRPPI_HAYAI_TESTS_SUPPORTED_EXECUTION_H
#define GRPPI_HAYAI_TESTS_SUPPORTED_EXECUTION_H


#include "seq/sequential_execution.h"
#include "native/parallel_execution_native.h"
#include "omp/parallel_execution_omp.h"
#include "tbb/parallel_execution_tbb.h"
#include "ff/parallel_execution_ff.h"
#include "dyn/dynamic_execution.h"

#include <time.h>

static inline void ticks_wait(long nanosec) {
	if (nanosec > 1000000L) nanosec =  1000000L;
	struct timespec req = {0, nanosec};
	nanosleep(&req, (struct timespec *)NULL);
}


#endif
