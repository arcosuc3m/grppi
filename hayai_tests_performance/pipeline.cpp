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

#include <hayai.hpp>

#include "pipeline.h"

#include "supported_executions.h"

#define SIZE 10000000
#define RUNS 10
#define ITRS 10
#define FILE "/space/Desk/GeneralNotes.txt" // path to input file

using namespace std;
using namespace grppi;

class PipelineTest {
	// Vectors
	ifstream file;

	int n;
	atomic<double> out = 0;

public:

	PipelineTest(int size=1000) : n{size} { }

	void setup_three_stages() {
	}

	void clear_multiple() {
	}

	void setup_vowels() {
		file.open(FILE, ifstream::in);
	}

	void clear_all() {
		file.close();
	}

	void run_three_stages(const dynamic_execution & e) {
		grppi::pipeline(e,
				[n]() -> optional<double> {
			static double i=0;
			if (++i<=n) return i;
			else return {};
		},
		[](double x) {
			return x*2;
		},
		[&](double x) {
			out += x;
		});
		cout << "[PIPE-3] Out is: " << out << endl;
	}

	void run_count_vowels(const dynamic_execution & e) {
		grppi::pipeline(e,
				[&file]() -> optional<string> {
			string word;
			file >> word;
			if (!file) { return {}; }
			else { return word; }
		},
		[](const string w) {
			string s = w;
			auto it = remove_if(begin(s), end(s),
					[](char c) {
				switch (c) {
				case 'a': case 'e': case 'i': case 'o': case 'u': return false;
				case 'A': case 'E': case 'I': case 'O': case 'U': return false;
				default: return true;
				}
			}
			);
			s.erase(it, end(s));
			return make_pair(w,s);
		},
		[](auto p) { return make_pair(p.first,p.second.length()); },
		[](auto p) {
			cout << p.first << " -> " << p.second << endl;
		}
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



class CountVowelsFixture : public ::hayai::Fixture {
public:
	virtual void SetUp() {
		this->test = new PipelineTest();
		test->setup_vowels();
	}

	virtual void TearDown() {
		test->clear_all();
		delete this->test;
	}

	PipelineTest* test;
};

BENCHMARK_F(CountVowelsFixture, pipe_count_vowels_seq, RUNS, ITRS) {
	sequential_execution seq;
	test->run_count_vowels(seq);
}

BENCHMARK_F(CountVowelsFixture, pipe_count_vowels_ff, RUNS, ITRS) {
	parallel_execution_ff ffexec;
	test->run_count_vowels(ffexec);
}

BENCHMARK_F(CountVowelsFixture, pipe_count_vowels_tbb, RUNS, ITRS) {
	parallel_execution_tbb tbbexec;
	test->run_count_vowels(tbbexec);
}

BENCHMARK_F(CountVowelsFixture, pipe_count_vowels_omp, RUNS, ITRS) {
	parallel_execution_omp ompexec;
	test->run_count_vowels(ompexec);
}

BENCHMARK_F(CountVowelsFixture, pipe_count_vowels_nat, RUNS, ITRS) {
	parallel_execution_native natexec;
	test->run_count_vowels(natexec);
}
