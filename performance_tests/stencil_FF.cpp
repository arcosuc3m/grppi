#include <iostream>
#include <streambuf>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>
#include <random>

#include <ff/parallel_for.hpp>

#include "supported_executions.h"

using namespace std;

class StencilTest {
	// Vectors
	vector<double> v;
	vector<double> v2;
	vector<double> w;
	long n;

public:

	StencilTest(long size=1000) : n{size} { }

	void setup_test() {
		double i=0.1, j=0.1;
		v.reserve(n); v2.reserve(n);
		w.reserve(n);
		for(long i=0; i<n; ++i) {
			v.push_back( static_cast<double>(i*3.14) );
			v2.push_back( static_cast<double>((2*v[i])/1.7) );
		}
	}

	void clear_all() {
		v.clear(); v2.clear();
		auto sum = std::accumulate(begin(w), end(w), 0.0);
		cerr << "Check sum is: " << sum << endl;
		w.clear();
	}

	// Stencil on two sequences.
	// w[i] <- v[i] + v[i-2] + v[i-1] + v[i+1] + v[i+2] +
	//         v2[i-2] + v2[i-1] + v2[i+1] + v2[i+2]
	void run_stencil_ff(int cores) {
		ff::ParallelFor pf(cores, true);

		// neigh_function
		auto vec_surronding = [](auto first, auto last, auto it) {
			vector<double> result;
			if (std::distance(first,it)>=2) result.push_back(*prev(it,2));
			if (std::distance(first,it)>=1) result.push_back(*prev(it,1));
			if (std::distance(it,last)>1)   result.push_back(*next(it,1));
			if (std::distance(it,last)>2)   result.push_back(*next(it,2));
			return result;
		};

		pf.parallel_for(0, n,
				[&](const long i) {
			auto r1 = vec_surronding(begin(v), end(v),v.begin()+i);
			auto r2 = vec_surronding(begin(v2), end(v2),v2.begin()+i);
			r1.insert(end(r1), begin(r2), end(r2));
			w[i] = v[i] + std::accumulate(begin(r1), end(r1),0.0);
		}, cores);
	}
};

int main(int argc, char **argv) {

	using namespace chrono;

	if(argc < 3) {
		cerr << "Usage: stencil_FF cores input_size" << endl;
		return -1;
	}

	int cores = stoi(argv[1]);
	int insize = stoi(argv[2]);

	StencilTest* test = new StencilTest(insize);
	test->setup_test();

	auto t1 = system_clock::now();
	test->run_stencil_ff(cores);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	test->clear_all();
	delete test;

	cerr << "[STENCIL-FF] Cores: " << cores
			<< "; InSize: " << insize
			<< "; Time: " << diff.count() << "ms"
			<< endl;
}
