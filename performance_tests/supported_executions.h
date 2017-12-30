#ifndef GRPPI_HAYAI_TESTS_SUPPORTED_EXECUTION_H
#define GRPPI_HAYAI_TESTS_SUPPORTED_EXECUTION_H


#include "seq/sequential_execution.h"
#include "native/parallel_execution_native.h"
#include "omp/parallel_execution_omp.h"
#include "tbb/parallel_execution_tbb.h"
#include "ff/parallel_execution_ff.h"
#include "dyn/dynamic_execution.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <time.h>

static inline void ticks_wait(long nanosec) {
	if (nanosec > 1000000L) nanosec =  1000000L;
	struct timespec req = {0, nanosec};
	nanosleep(&req, (struct timespec *)NULL);
}

int insize[] = {10000, 100000, 1000000};

namespace { // -- utility functions

inline void run_cmd(const char *cmd, std::string& result) {
	std::array<char, 128> buffer;
	std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);

	if (!pipe) throw std::runtime_error("popen() failed!");

	result.clear();
	while (!feof(pipe.get()))
		if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
			result += buffer.data();
}

inline int get_phys_cores() {
	int count=1, cs=0, sk=0;
	std::string res{};

#if defined(__linux__)
	char cores[] = "lscpu | grep 'Core(s)' | awk '{print $4}'";
	char socks[] = "lscpu | grep 'Socket(s)' | awk '{print $2}'";

	run_cmd(cores, res);
	cs = std::atoi(res.c_str());

	run_cmd(socks, res);
	sk = std::atoi(res.c_str());

	count = cs*sk;

#elif defined (__APPLE__)
	char cmd[] = "sysctl hw.physicalcpu | awk '{print $2}'";

	run_cmd(cmd, res);
	count = std::stoi(res);

#else
#pragma message ("Cannot determine physical cores number on this platform")
#endif

	return count;
}

} // Anonymous namespace


#endif
