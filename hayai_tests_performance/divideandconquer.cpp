/*
 * map.cpp
 *
 *  Created on: 21 Oct 2017
 *      Author: fabio
 */



#include <iostream>
#include <streambuf>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>
#include <random>

#include <hayai.hpp>

#include "divideconquer.h"

#include "supported_executions.h"

#define SIZE 10000000
#define RUNS 10
#define ITRS 10

using namespace std;
using namespace grppi;

class DACTest {
	// Vectors
	vector<double> v{};

	int n;

	random_device rengine;
	uniform_real_distribution<double> value_gen{-100.4, 100.7};
	uniform_real_distribution<double> coef_gen{1.4, 10.7};
	double a = coef_gen(rengine);

public:

	DACTest(int size=1000) : n{size} { }

	void setup_unary() {
		v.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void clear_unary() {
		v.clear();
	}

	// run_vecsum_chunked
	auto run_vecsum_chunked(const dynamic_execution & e) {
		return grppi::divide_conquer(e, v,
				[this](auto & v) { // Divide
			std::vector<std::vector<double>> subproblem;
			auto mid1 = std::next(v.begin(), v.size()/3);
			auto mid2 = std::next(v.begin(), 2*v.size()/3);
			subproblem.push_back({v.begin(), mid1});
			subproblem.push_back({mid1,mid2});
			subproblem.push_back({mid2, v.end()});
			return subproblem;
		},
		// predicate
		[this](auto x) {
			return x.size()>3;
		},
		// Solve base case
		[this](auto problem) {
			return std::accumulate(problem.begin(), problem.end(), 0);
		},
		// Combine
		[this](auto  p1, auto  p2) {
			return p1 + p2;
		});
	}
};

// -- Define fixtures for tests

class DACFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new DACTest(SIZE);
		test->setup_unary();
	}

	virtual void TearDown() {
		test->clear_unary();
		delete this->test;
	}

	DACTest* test;
};

BENCHMARK_F(DACFixture, dac_vecsum_seq, RUNS, ITRS) {
	sequential_execution sequq;
	test->run_vecsum_chunked(sequq);
}

BENCHMARK_F(DACFixture, dac_vecsum_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_vecsum_chunked(ffexec);
}

BENCHMARK_F(DACFixture, dac_vecsum_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_vecsum_chunked(tbbexec);
}

BENCHMARK_F(DACFixture, dac_vecsum_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_vecsum_chunked(ompexec);
}

BENCHMARK_F(DACFixture, dac_vecsum_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_vecsum_chunked(natexec);
}
