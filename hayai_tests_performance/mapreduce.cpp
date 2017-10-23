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

#define SIZE 10000000
#define RUNS 10
#define ITRS 10
#define FILE "/space/Desk/GeneralNotes.txt" // path to input file


using namespace std;
using namespace grppi;

class MapReduceTest {
	// Vectors
	vector<double> v;
	vector<string> words;

	int n;

	random_device rengine;
	uniform_real_distribution<double> value_gen{-100.4, 100.7};
	uniform_real_distribution<double> coef_gen{1.4, 10.7};
	double a = coef_gen(rengine);

public:

	MapReduceTest(int size=1000) : n{size} { }

	void setup_square_sum() {
		v.reserve(n);
		generate_n(back_inserter(v), n,
				[&]() { return value_gen(rengine); });
	}

	void setup_word_count() {
		ifstream file(FILE);
		copy(istream_iterator<string>{file}, istream_iterator<string>{},
				back_inserter(words));
		file.close();
	}

	void clear_square_sum() {
		v.clear();
	}

	void clear_word() {
		words.clear();
	}

	auto run_square_sum(const dynamic_execution & e) {
		return grppi::map_reduce(e, v.begin(), v.end(), 0,
				[&](double x) { return x*2; },
				[](double x, double y) { return x+y; }
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

// -- Define fixtures for tests

class SquareSumFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new MapReduceTest(SIZE);
		test->setup_square_sum();
	}

	virtual void TearDown() {
		test->clear_square_sum();
		delete this->test;
	}

	MapReduceTest* test;
};

BENCHMARK_F(SquareSumFixture, square_sum_seq, RUNS, ITRS) {
	sequential_execution sequq;
	test->run_square_sum(sequq);
}

BENCHMARK_F(SquareSumFixture, square_sum_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_square_sum(ffexec);
}

BENCHMARK_F(SquareSumFixture, square_sum_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_square_sum(tbbexec);
}

BENCHMARK_F(SquareSumFixture, square_sum_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_square_sum(ompexec);
}

BENCHMARK_F(SquareSumFixture, square_sum_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_square_sum(natexec);
}

// words count
class WordsCountFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new MapReduceTest();
		test->setup_word_count();
	}

	virtual void TearDown() {
		test->clear_word();
		delete this->test;
	}

	MapReduceTest* test;
};

BENCHMARK_F(WordsCountFixture, words_count_seq, RUNS, ITRS) {
	sequential_execution sequq;
	test->run_square_sum(sequq);
}

BENCHMARK_F(WordsCountFixture, words_count_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_square_sum(ffexec);
}

BENCHMARK_F(WordsCountFixture, words_count_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_square_sum(tbbexec);
}

BENCHMARK_F(WordsCountFixture, words_count_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_square_sum(ompexec);
}

BENCHMARK_F(WordsCountFixture, words_count_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_square_sum(natexec);
}
