/*
 * map.cpp
 *
 *  Created on: 21 Oct 2017
 *      Author: fabio
 */


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

#include <hayai.hpp>

#include "farm.h"
#include "pipeline.h"

#include "supported_executions.h"

#define SIZE 10000000
#define RUNS 10
#define ITRS 10

using namespace std;
using namespace grppi;

class FarmTest {
	// Vectors
	vector<double> v;
//	vector<double> v2;
//	vector<double> v3;
	vector<double> w;
	atomic<double> output;

	int n;
	atomic<int> idx_in;
	atomic<int> idx_out;

	random_device rengine;
	uniform_real_distribution<double> value_gen{-100.4, 100.7};
	uniform_real_distribution<double> coef_gen{1.4, 10.7};
	double *a;

	void do_work(double * task, double val) {
		for(register int i=0;i<n;++i)
			task[i]+=val*a[i];
		ticks_wait(1000L);
	}

public:

	FarmTest(int size=1000) : n{size} {
		a = new double[n];
		for(int i=0;i<n;++i)
			a[i] = coef_gen(rengine);
	}

	void setup_multiple() {
		w.reserve(n);
		v.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void clear_multiple() {
		v.clear(); w.clear();
	}

//	void setup_ary() {
//		v.reserve(n); v2.reserve(n);
//		v3.reserve(n); w.reserve(n);
//		generate_n(back_inserter(v), n,
//				[&]() { return value_gen(rengine); });
//	}
//
//	void clear_all() {
//		v.clear(); v2.clear();
//		v3.clear(); w.clear();
//	}


	void run_multiple(const dynamic_execution & e) {
		grppi::pipeline(e,
				[&]() -> experimental::optional<double> {
			if (idx_in < v.size()) {
				idx_in++;
				return v[idx_in-1];
			} else
				return {};
		},
		grppi::farm(4,
				[&](double x) {
			double * task = new double[n]();
			do_work(task, x); // basically waste time and do some math
			delete[] task;
			return x*2.0;
		}),
		[&](double x) {
			w[idx_out] = x;
			idx_out++;
		}
		);
	}
};

// -- Define fixtures for tests
class FarmFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new FarmTest(SIZE);
		test->setup_multiple();
	}

	virtual void TearDown() {
		test->clear_multiple();
		delete this->test;
	}

	FarmTest* test;
};

BENCHMARK_F(FarmFixture, farm_multiple_seq, RUNS, ITRS) {
	sequential_execution seq;
	test->run_multiple(seq);
}

BENCHMARK_F(FarmFixture, farm_multiple_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_multiple(ffexec);
}

BENCHMARK_F(FarmFixture, farm_multiple_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_multiple(tbbexec);
}

BENCHMARK_F(FarmFixture, farm_multiple_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_multiple(ompexec);
}

BENCHMARK_F(FarmFixture, farm_multiple_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_multiple(natexec);
}



//class FarmAryFixture : public ::hayai::Fixture {
//public:
//	virtual void SetUp() {
//		this->maptest =
//				new MapTests(1000000);
//		maptest->setup_daxpy();
//	}
//
//	virtual void TearDown() {
//		maptest->clear_all();
//		delete this->maptest;
//	}
//
//	MapTests* maptest;
//};
//
//
//
//BENCHMARK_F(FarmAryFixture, farm_daxpy_seq, RUNS, ITRS) {
//	sequential_execution seq;
//	maptest->run_daxpy(seq);
//}
//
//BENCHMARK_F(FarmAryFixture, farm_daxpy_ff, RUNS, ITRS) {
//	parallel_execution_ff ffexec;
//	maptest->run_daxpy(ffexec);
//}
//
//BENCHMARK_F(FarmAryFixture, farm_daxpy_tbb, RUNS, ITRS) {
//	parallel_execution_tbb tbbexec;
//	maptest->run_daxpy(tbbexec);
//}
//
//BENCHMARK_F(FarmAryFixture, farm_daxpy_omp, RUNS, ITRS) {
//	parallel_execution_omp ompexec;
//	maptest->run_daxpy(ompexec);
//}
//
//BENCHMARK_F(FarmAryFixture, farm_daxpy_nat, RUNS, ITRS) {
//	parallel_execution_native natexec;
//	maptest->run_daxpy(natexec);
//}
