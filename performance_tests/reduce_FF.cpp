#include <atomic>
#include <iostream>
#include <streambuf>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>
#include <random>
#include <experimental/optional>

#include <ff/parallel_for.hpp>
#include "supported_executions.h"

using namespace std;

class ReduceTest {
	// Vectors
	vector<double> v;
	long n;
	double a;

public:

	ReduceTest(long size=1000) : n{size}, a(3.7573) {

	}

	void setup_reduce() {
		v.reserve(n);
		for(long i=0; i<n; ++i) {
			v.push_back( static_cast<double>(i*3.14) );
		}
	}

	void clear_all() {
		v.clear();
	}

	auto run_sum_ff(int cores) {
		ff::ParallelForReduce<double> pfr(cores, true);

		double vaR = 0, varR_zero= 0;

		auto final_red = [&](double &a, const double &r) {
			a += r;
		};

		pfr.parallel_reduce(vaR, varR_zero, 0, n,
				[&](const long i, double &vaR) {
			vaR += v[i];
		}, final_red, cores);

		return vaR;
	}
};

int main(int argc, char **argv) {

	using namespace chrono;

	if(argc < 3) {
		cerr << "Usage: reduce_FF cores input_size" << endl;
		return -1;
	}

	int cores = stol(argv[1]);
	long insize = stol(argv[2]);

	ReduceTest* test = new ReduceTest(insize);
	test->setup_reduce();

	auto t1 = system_clock::now();
	auto res = test->run_sum_ff(cores);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	test->clear_all();
	delete test;

	cerr << "Result is: " << res << endl;

	cerr << "[RED-FF] Cores: " << cores
				<< "; InSize: " << insize
				<< "; Time: " << diff.count() << "ms"
				<< endl;
}
