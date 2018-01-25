#ifndef GRPPI_PERFORMANCE_TESTS_H
#define GRPPI_PERFORMANCE_TESTS_H

#include "dyn/dynamic_execution.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <time.h>

grppi::dynamic_execution execution_mode(const std::string & opt, int conc_degr=0, bool order=true) {
  using namespace grppi;
  if ("seq" == opt) return sequential_execution();
  if ("thr" == opt) { 
  	if(conc_degr != 0)
  		return parallel_execution_native(conc_degr, order);
  	else {
  		parallel_execution_native e{};
  		if(!order)
  			e.disable_ordering();
  		return e;
  	}
  }
  if ("omp" == opt) { 
  	if(conc_degr != 0)
  		return parallel_execution_omp(conc_degr, order);
  	else {
  		parallel_execution_omp e{};
  		if(!order)
  			e.disable_ordering();
  		return e;
  	}
  }
  if ("tbb" == opt) { 
  	if(conc_degr != 0)
  		return parallel_execution_tbb(conc_degr, order);
  	else {
  		parallel_execution_tbb e{};
  		if(!order)
  			e.disable_ordering();
  		return e;
  	}
  }
  if ("ff" == opt) { 
  	if(conc_degr != 0)
  		return parallel_execution_ff(conc_degr, order);
  	else {
  		parallel_execution_ff e{};
  		if(!order)
  			e.disable_ordering();
  		return e;
  	}
  }
  return {};
}

void print_available_modes(std::ostream & os) {
  using namespace std;
  using namespace grppi;

  if (is_supported<sequential_execution>()) {
    os << "    seq -> Sequential execution" << endl;
  }

  if (is_supported<parallel_execution_native>()) {
    os << "    thr -> ISO C++ Threads backend" << endl;
  }

  if (is_supported<parallel_execution_tbb>()) {
    os << "    tbb -> Intel TBB backend" << endl;
  }

  if (is_supported<parallel_execution_omp>()) {
    os << "    omp -> OpenMP backend" << endl;
  }

  if (is_supported<parallel_execution_ff>()) {
    os << "    ff  -> FastFlow backend" << endl;
  }
}

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
	cs = std::stoi(res);

	run_cmd(socks, res);
	sk = std::stoi(res);

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

