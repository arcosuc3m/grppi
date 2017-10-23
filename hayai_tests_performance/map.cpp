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

using namespace std;
using namespace grppi;

class MapTests {
	//T execution_;

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

	void setup_double() {
		w.reserve(n);
		v.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void setup_daxpy() {
		v.reserve(n); v2.reserve(n);
		v3.reserve(n); w.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	MapTests(int size=10000000) : n{size} { }

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

// -- Define fixtures for map tests
class DoubleFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->maptest =	new MapTests(1000000);
		maptest->setup_double();
	}

	virtual void TearDown() {
		maptest->clear_double();
		delete this->maptest;
	}

	MapTests* maptest;
};


class DaxpyFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->maptest =
				new MapTests(1000000);
		maptest->setup_daxpy();
	}

	virtual void TearDown() {
		maptest->clear_all();
		delete this->maptest;
	}

	MapTests* maptest;
};

BENCHMARK_F(DoubleFixture, map_double_seq, 10, 10) {
	sequential_execution seq;
	maptest->run_double(seq);
}

BENCHMARK_F(DaxpyFixture, map_daxpy_seq, 10, 10) {
	sequential_execution seq;
	maptest->run_daxpy(seq);
}

BENCHMARK_F(DoubleFixture, map_double_ff, 10, 10) {
	parallel_execution_ff ffexec;
	maptest->run_double(ffexec);
}

BENCHMARK_F(DaxpyFixture, map_daxpy_ff, 10, 10) {
	parallel_execution_ff ffexec;
	maptest->run_daxpy(ffexec);
}

BENCHMARK_F(DoubleFixture, map_double_tbb, 10, 10) {
	parallel_execution_tbb tbbexec;
	maptest->run_double(tbbexec);
}

BENCHMARK_F(DaxpyFixture, map_daxpy_tbb, 10, 10) {
	parallel_execution_tbb tbbexec;
	maptest->run_daxpy(tbbexec);
}

BENCHMARK_F(DoubleFixture, map_double_omp, 10, 10) {
	parallel_execution_omp ompexec;
	maptest->run_double(ompexec);
}

BENCHMARK_F(DaxpyFixture, map_daxpy_omp, 10, 10) {
	parallel_execution_omp ompexec;
	maptest->run_daxpy(ompexec);
}

BENCHMARK_F(DoubleFixture, map_double_nat, 10, 10) {
	parallel_execution_native natexec;
	maptest->run_double(natexec);
}

BENCHMARK_F(DaxpyFixture, map_daxpy_nat, 10, 10) {
	parallel_execution_native natexec;
	maptest->run_daxpy(natexec);
}
