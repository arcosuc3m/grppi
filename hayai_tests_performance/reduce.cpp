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

#include "reduce.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

class TestClass {
	//T execution_;

	// Vectors
	vector<double> v;

	int n;

	random_device rengine;
	uniform_real_distribution<double> value_gen{-100.4, 100.7};
	uniform_real_distribution<double> coef_gen{1.4, 10.7};
	double a = coef_gen(rengine);

public:

	TestClass(int size=10000000) : n{size} { }

	void setup_reduce() {
		v.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void clear_reduce() {
		v.clear();
	}

	auto run_square_sum(const dynamic_execution & e) {
		return grppi::reduce(e, begin(v), end(v), 0.0,
				[&](auto x, auto y) { return x + y; }
		);
	}
};

// -- Define fixtures for map tests

class ReduceFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->maptest =	new TestClass();
		maptest->setup_reduce();
	}

	virtual void TearDown() {
		maptest->clear_reduce();
		delete this->maptest;
	}

	TestClass* maptest;
};

BENCHMARK_F(ReduceFixture, reduce_seq, 10, 10) {
	sequential_execution sequq;
	maptest->run_square_sum(sequq);
}

BENCHMARK_F(ReduceFixture, reduce_ff, 10, 10) {
	parallel_execution_ff ffexec;
	maptest->run_square_sum(ffexec);
}

BENCHMARK_F(ReduceFixture, reduce_tbb, 10, 10) {
	parallel_execution_tbb tbbexec;
	maptest->run_square_sum(tbbexec);
}

BENCHMARK_F(ReduceFixture, reduce_omp, 10, 10) {
	parallel_execution_omp ompexec;
	maptest->run_square_sum(ompexec);
}

BENCHMARK_F(ReduceFixture, reduce_nat, 10, 10) {
	parallel_execution_native natexec;
	maptest->run_square_sum(natexec);
}
