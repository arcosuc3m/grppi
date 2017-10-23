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
#include <utility>
#include <experimental/optional>

#include <hayai.hpp>

#include "pipeline.h"

#include "supported_executions.h"

#define SIZE 10000
#define RUNS 10
#define ITRS 10
#define FILE "/space/Desk/GeneralNotes.txt" // path to input file

using namespace std;
using namespace grppi;

template <typename T>
using optional = std::experimental::optional<T>;

class PipelineTest {
	// Vectors
	map<string,size_t> vowels;

	int n;
	atomic<long long> out;

public:

	PipelineTest(int size=1000) : n{size} { }

	void setup_three_stages() {
	}

	void clear_multiple() {
	}

	void setup_vowels() {
	}

	void clear_all() {
	}

	void run_three_stages(const dynamic_execution & e) {
		grppi::pipeline(e,
				[&,i=0]() mutable -> optional<long long> {
			if (++i<=n) return i;
			else return {};
		},
		[](auto x) {
			return x*2;
		},
		[&](auto x) {
			out += x;
		});
	}

//	void run_count_vowels(const dynamic_execution & e) {
//		grppi::pipeline(e,
//				[&]() -> experimental::optional<string> {
//			string word;
//			file >> word;
//			if (!file) { return {}; }
//			else { return word; }
//		},
//		[](const string w) {
//			string s = w;
//			auto it = remove_if(begin(s), end(s),
//					[](char c) {
//				switch (c) {
//				case 'a': case 'e': case 'i': case 'o': case 'u': return false;
//				default: return true;
//				}
//			});
//			s.erase(it, end(s));
//			return make_pair(w,s);
//		},
//		[](auto p) { return make_pair(p.first,p.second.length()); },
//	    [](auto p) {
//	      //cout << p.first << " -> " << p.second << endl;
//			vowels.insert(std::move(p));
//		}
//		);
//	}

	void run_composed_piecewise(const dynamic_execution & e, size_t n) {
		auto inner = grppi::pipeline(
				[this](int x) { return x*x; },
				[this](int x) {	return x+1;	}
		);

		grppi::pipeline(e,
				[this,&n,i=0]() mutable -> optional<int> {
			if (++i<=n) return i;
			else return {};
		},
		//inner,
		grppi::pipeline(
				[this](int x) {	return x*x;	},
				[this](int x) {	return x+1; }
		),
		[this](int x) {	out += x; }
		);
	}
};

// -- Define fixtures for tests
class PipelineFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new PipelineTest(SIZE);
	}

	virtual void TearDown() {
		delete this->test;
	}

	PipelineTest* test;
};

BENCHMARK_F(PipelineFixture, pipe_three_stages_seq, RUNS, ITRS) {
	sequential_execution seq;
	test->run_three_stages(seq);
}

BENCHMARK_F(PipelineFixture, pipe_three_stages_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_three_stages(ffexec);
}

BENCHMARK_F(PipelineFixture, pipe_three_stages_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_three_stages(tbbexec);
}

BENCHMARK_F(PipelineFixture, pipe_three_stages_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_three_stages(ompexec);
}

BENCHMARK_F(PipelineFixture, pipe_three_stages_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_three_stages(natexec);
}



class PipeComposedFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new PipelineTest(SIZE);
		test->setup_vowels();
	}

	virtual void TearDown() {
		test->clear_all();
		delete this->test;
	}

	PipelineTest* test;
};

BENCHMARK_F(PipeComposedFixture, pipe_count_vowels_seq, RUNS, ITRS) {
	sequential_execution seq;
	test->run_composed_piecewise(seq, SIZE);
}

BENCHMARK_F(PipeComposedFixture, pipe_count_vowels_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_composed_piecewise(ffexec, SIZE);
}

BENCHMARK_F(PipeComposedFixture, pipe_count_vowels_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_composed_piecewise(tbbexec, SIZE);
}

BENCHMARK_F(PipeComposedFixture, pipe_count_vowels_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_composed_piecewise(ompexec, SIZE);
}

BENCHMARK_F(PipeComposedFixture, pipe_count_vowels_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_composed_piecewise(natexec, SIZE);
}
