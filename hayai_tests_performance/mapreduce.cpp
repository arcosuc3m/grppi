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

#include "mapreduce.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

class TestClass {
	//T execution_;

	// Vectors
	vector<double> v;
	vector<string> words;
//	vector<double> v3;
//	vector<double> w;

	int n;

	random_device rengine;
	uniform_real_distribution<double> value_gen{-100.4, 100.7};
	uniform_real_distribution<double> coef_gen{1.4, 10.7};
	double a = coef_gen(rengine);

public:

	TestClass(int size=10000000) : n{size} { }

	void setup_square_sum() {
		v.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void setup_word_count() {
		// change here the path to input file
		ifstream file("/space/Desk/GeneralNotes.txt");
		//v2.reserve(n); v3.reserve(n); w.reserve(n);
		copy(istream_iterator<string>{file}, istream_iterator<string>{},
				back_inserter(words));
	}

	void clear_word() {
		words.clear();
	}

	void clear_square_sum() {
		v.clear();
	}

	auto run_square_sum(const dynamic_execution & e) {
		return grppi::map_reduce(e, v.begin(), v.end(), 0,
				[this](double x) { return x*2; },
				[](double x, double y) { return x + y; }
				);
	}

	auto run_word_count(const dynamic_execution & ex) {
		return grppi::map_reduce(ex,
				words.begin(), words.end(), map<string,int>{},
				[](string word) -> map<string,int> { return {{word,1}}; },
				[](map<string,int> & lhs, const map<string,int> & rhs) -> map<string,int> & {
					for (auto & w : rhs) {
						lhs[w.first]+= w.second;
					}
					return lhs;
				}
		);
	}
};

// -- Define fixtures for map tests

class SquareSumFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->maptest =	new TestClass();
		maptest->setup_square_sum();
	}

	virtual void TearDown() {
		maptest->clear_square_sum();
		delete this->maptest;
	}

	TestClass* maptest;
};

class WordClassFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->maptest =	new TestClass();
		maptest->setup_word_count();
	}

	virtual void TearDown() {
		maptest->clear_word();
		delete this->maptest;
	}

	TestClass* maptest;
};

BENCHMARK_F(SquareSumFixture, square_sum_seq, 10, 10) {
	sequential_execution sequq;
	maptest->run_square_sum(sequq);
}

BENCHMARK_F(SquareSumFixture, square_sum_ff, 10, 10) {
	parallel_execution_ff ffexec;
	maptest->run_square_sum(ffexec);
}

BENCHMARK_F(SquareSumFixture, square_sum_tbb, 10, 10) {
	parallel_execution_tbb tbbexec;
	maptest->run_square_sum(tbbexec);
}

BENCHMARK_F(SquareSumFixture, square_sum_omp, 10, 10) {
	parallel_execution_omp ompexec;
	maptest->run_square_sum(ompexec);
}

BENCHMARK_F(SquareSumFixture, square_sum_nat, 10, 10) {
	parallel_execution_native natexec;
	maptest->run_square_sum(natexec);
}

// word count

BENCHMARK_F(WordClassFixture, words_count_seq, 10, 10) {
	sequential_execution sequq;
	maptest->run_square_sum(sequq);
}

BENCHMARK_F(WordClassFixture, words_count_ff, 10, 10) {
	parallel_execution_ff ffexec;
	maptest->run_square_sum(ffexec);
}

BENCHMARK_F(WordClassFixture, words_count_tbb, 10, 10) {
	parallel_execution_tbb tbbexec;
	maptest->run_square_sum(tbbexec);
}

BENCHMARK_F(WordClassFixture, words_count_omp, 10, 10) {
	parallel_execution_omp ompexec;
	maptest->run_square_sum(ompexec);
}

BENCHMARK_F(WordClassFixture, words_count_nat, 10, 10) {
	parallel_execution_native natexec;
	maptest->run_square_sum(natexec);
}
