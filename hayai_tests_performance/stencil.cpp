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

#include "stencil.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

class TestClass {
	//T execution_;

	// Vectors
	vector<double> v{};
	vector<double> v2{};
	vector<double> w{};

	int n;

	random_device rengine;
	uniform_real_distribution<double> value_gen{-100.4, 100.7};
	uniform_real_distribution<double> coef_gen{1.4, 10.7};
	double a = coef_gen(rengine);

public:

	TestClass(int size=10000000) : n{size} { }

	void setup_unary() {
		v.reserve(n); w.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void clear_unary() {
		v.clear(); w.clear();
	}

	void setup_nary() {
		v.reserve(n); v2.reserve(n); w.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
		generate_n(back_inserter(v2), n,
				[&]() { return value_gen(rengine); });
	}

	void clear_nary() {
		v.clear(); w.clear(); v2.clear();
	}

	// Stencil on a single sequence.
	// Every v[i] is assigned v[i] + v[i+1] or v[i] + 0 if v[i+1] does not exist
	void run_unary(const dynamic_execution & ex) {
		grppi::stencil(ex, begin(v), end(v), begin(w),
				[this](auto it, auto n) {
			return *it + n;
		},
		[&](auto it) {
			if (it+1 != v.end())
				return *(it+1);
			else
				return 0.0;
		}
		);
	}

	// Stencil on two sequences.
	// Each v[i] is made
	//     v[i-2] + v[i-1] + v[i+1] + v[i+2] +
	//     w[i-2] + w[i-1] + w[i+1] + w[i+2] +
	void run_nary(const dynamic_execution & ex) {
		auto vec_surronding = [](auto first, auto last, auto it) {
			vector<int> result;
			if (distance(first,it)>=2) result.push_back(*prev(it,2));
			if (distance(first,it)>=1) result.push_back(*prev(it,1));
			if (distance(it,last)>1) result.push_back(*next(it,1));
			if (distance(it,last)>2) result.push_back(*next(it,2));
			return result;
		};

		grppi::stencil(ex, begin(v), end(v), begin(w),
				// Stencil computes average of neighbours
				[this](auto it, const auto & n) {
			return std::accumulate(begin(n),end(n),0);
		},
		// Neighbours are i-2, i-1, i+1, i+2 of currrent position
		[&,this](auto it, auto it2) {
			auto r1 = vec_surronding(begin(v), end(v), it);
			auto r2 = vec_surronding(begin(v2), end(v2), it2);
			r1.insert(end(r1), begin(r2), end(r2));
			return r1;
		},
		begin(v2)
		);
	}
};

// -- Define fixtures for map tests

class StencilUnaryFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->maptest =	new TestClass();
		maptest->setup_unary();
	}

	virtual void TearDown() {
		maptest->clear_unary();
		delete this->maptest;
	}

	TestClass* maptest;
};

BENCHMARK_F(StencilUnaryFixture, stencil_unary_seq, 10, 10) {
	sequential_execution sequq;
	maptest->run_unary(sequq);
}

BENCHMARK_F(StencilUnaryFixture, stencil_unary_ff, 10, 10) {
	parallel_execution_ff ffexec;
	maptest->run_unary(ffexec);
}

BENCHMARK_F(StencilUnaryFixture, stencil_unary_tbb, 10, 10) {
	parallel_execution_tbb tbbexec;
	maptest->run_unary(tbbexec);
}

BENCHMARK_F(StencilUnaryFixture, stencil_unary_omp, 10, 10) {
	parallel_execution_omp ompexec;
	maptest->run_unary(ompexec);
}

BENCHMARK_F(StencilUnaryFixture, stencil_unary_nat, 10, 10) {
	parallel_execution_native natexec;
	maptest->run_unary(natexec);
}

// nary stencil

class StencilNaryFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->maptest =	new TestClass();
		maptest->setup_unary();
	}

	virtual void TearDown() {
		maptest->clear_unary();
		delete this->maptest;
	}

	TestClass* maptest;
};

BENCHMARK_F(StencilNaryFixture, stencil_nary_seq, 10, 10) {
	sequential_execution sequq;
	maptest->run_nary(sequq);
}

BENCHMARK_F(StencilNaryFixture, stencil_nary_ff, 10, 10) {
	parallel_execution_ff ffexec;
	maptest->run_nary(ffexec);
}

BENCHMARK_F(StencilNaryFixture, stencil_nary_tbb, 10, 10) {
	parallel_execution_tbb tbbexec;
	maptest->run_nary(tbbexec);
}

BENCHMARK_F(StencilNaryFixture, stencil_nary_omp, 10, 10) {
	parallel_execution_omp ompexec;
	maptest->run_nary(ompexec);
}

BENCHMARK_F(StencilNaryFixture, stencil_nary_nat, 10, 10) {
	parallel_execution_native natexec;
	maptest->run_nary(natexec);
}
