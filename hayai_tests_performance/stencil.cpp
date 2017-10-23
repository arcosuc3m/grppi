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

#define SIZE 10000000
#define RUNS 10
#define ITRS 10

using namespace std;
using namespace grppi;

class StencilTest {
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

	StencilTest(int size=1000) : n{size} { }

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
	// w[i] <- v[i] + v[i+1] or v[i] + 0 if v[i+1] does not exist
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
	// w[i] <- v[i-2] + v[i-1] + v[i+1] + v[i+2] +
	//         v2[i-2] + v2[i-1] + v2[i+1] + v2[i+2]
	void run_nary(const dynamic_execution & ex) {
		auto vec_surronding = [](auto first, auto last, auto it) {
			vector<double> result;
			if (std::distance(first,it)>=2) result.push_back(*prev(it,2));
			if (std::distance(first,it)>=1) result.push_back(*prev(it,1));
			if (std::distance(it,last)>1)   result.push_back(*next(it,1));
			if (std::distance(it,last)>2)   result.push_back(*next(it,2));
			return result;
		};

		grppi::stencil(ex, begin(v), end(v), begin(w),
				[this](auto it, const auto & n) {
			return std::accumulate(begin(n),end(n),0);
		},
		[&](auto it, auto it2) {
			auto r1 = vec_surronding(begin(v), end(v), it);
			auto r2 = vec_surronding(begin(v2), end(v2), it2);
			r1.insert(end(r1), begin(r2), end(r2));
			return r1;
		},
		begin(v2)
		);
	}
};

// -- Define fixtures for tests

class StencilUnaryFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new StencilTest(SIZE);
		test->setup_unary();
	}

	virtual void TearDown() {
		test->clear_unary();
		delete this->test;
	}

	StencilTest* test;
};

BENCHMARK_F(StencilUnaryFixture, stencil_unary_seq, RUNS, ITRS) {
	sequential_execution sequq;
	test->run_unary(sequq);
}

BENCHMARK_F(StencilUnaryFixture, stencil_unary_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_unary(ffexec);
}

BENCHMARK_F(StencilUnaryFixture, stencil_unary_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_unary(tbbexec);
}

BENCHMARK_F(StencilUnaryFixture, stencil_unary_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_unary(ompexec);
}

BENCHMARK_F(StencilUnaryFixture, stencil_unary_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_unary(natexec);
}

// nary stencil

class StencilNaryFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test =	new StencilTest(SIZE);
		test->setup_nary();
	}

	virtual void TearDown() {
		test->clear_nary();
		delete this->test;
	}

	StencilTest* test;
};

BENCHMARK_F(StencilNaryFixture, stencil_nary_seq, RUNS, ITRS) {
	sequential_execution sequq;
	test->run_nary(sequq);
}

BENCHMARK_F(StencilNaryFixture, stencil_nary_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_nary(ffexec);
}

BENCHMARK_F(StencilNaryFixture, stencil_nary_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_nary(tbbexec);
}

BENCHMARK_F(StencilNaryFixture, stencil_nary_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_nary(ompexec);
}

BENCHMARK_F(StencilNaryFixture, stencil_nary_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_nary(natexec);
}
