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

#include "map.h"

#include "supported_executions.h"

#define SIZE 1000000
#define RUNS 10
#define ITRS 10

using namespace std;
using namespace grppi;

class MapTest {
	// Vectors
	vector<double> v;
	vector<double> v2;
	vector<double> v3;
	vector<double> w;

	int n;

	random_device rengine;
	uniform_real_distribution<double> value_gen{-100.4, 100.7};
	uniform_real_distribution<double> coef_gen{1.4, 10.7};
	double a = coef_gen(rengine);

public:

	MapTest(int size=1000) : n{size} { }

	void setup_double() {
		w.reserve(n); v.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void setup_daxpy() {
		v.reserve(n); v2.reserve(n);
		v3.reserve(n); w.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void clear_double() {
		v.clear(); w.clear();
	}

	void clear_all() {
		v.clear(); v2.clear();
		v3.clear(); w.clear();
	}

	void run_double(const dynamic_execution & exec) {
		grppi::map(exec,
				v.begin(), v.end(), w.begin(),
				[this](double i) {
			return i*2.0;
		}
		);
	}

	void run_daxpy(const dynamic_execution & exec) {
		grppi::map(exec,
				v.begin(), v.end(), w.begin(),
				[this](double x, double y, double z) {
			return this->a*x+y+z;
		},
		v2.begin(), v3.begin()
		);
	}
};


// -- Define fixtures for tests

//class MapDoubleFixture : public ::hayai::Fixture {
//public:
//	virtual void SetUp() {
//		this->test = new MapTest(SIZE);
//		test->setup_double();
//	}
//
//	virtual void TearDown() {
//		test->clear_double();
//		delete this->test;
//	}
//
//	MapTest* test;
//};
//
//BENCHMARK_F(MapDoubleFixture, map_double_seq, RUNS, ITRS) {
//	sequential_execution seq;
//	test->run_double(seq);
//}
//
//BENCHMARK_F(MapDoubleFixture, map_double_ff, RUNS, ITRS) {
//	parallel_execution_ff ffexec;
//	test->run_double(ffexec);
//}
//
//BENCHMARK_F(MapDoubleFixture, map_double_tbb, RUNS, ITRS) {
//	parallel_execution_tbb tbbexec;
//	test->run_double(tbbexec);
//}
//
//BENCHMARK_F(MapDoubleFixture, map_double_omp, RUNS, ITRS) {
//	parallel_execution_omp ompexec;
//	test->run_double(ompexec);
//}
//
//BENCHMARK_F(MapDoubleFixture, map_double_nat, RUNS, ITRS) {
//	parallel_execution_native natexec;
//	test->run_double(natexec);
//}


class MapDaxpyFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new MapTest(SIZE);
		test->setup_daxpy();
	}

	virtual void TearDown() {
		test->clear_all();
		delete this->test;
	}

	MapTest* test;
};


BENCHMARK_F(MapDaxpyFixture, map_daxpy_seq, RUNS, ITRS) {
	sequential_execution seq;
	test->run_daxpy(seq);
}

BENCHMARK_F(MapDaxpyFixture, map_daxpy_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_daxpy(ffexec);
}

BENCHMARK_F(MapDaxpyFixture, map_daxpy_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_daxpy(tbbexec);
}

BENCHMARK_F(MapDaxpyFixture, map_daxpy_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_daxpy(ompexec);
}

BENCHMARK_F(MapDaxpyFixture, map_daxpy_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_daxpy(natexec);
}
