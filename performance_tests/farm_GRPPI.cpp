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

#include "farm.h"
#include "pipeline.h"
#include "supported_executions.h"

using namespace std;
using namespace grppi;


class FarmTest {
	// Vectors
	vector<long> v;
	long output;
	long idx_in;
	long n;

	long do_work(long val) {
	  for(register int i=0;i<1000;++i){
	    val += i;
	  }
	  return val;
	}

public:

	FarmTest(long size) : n{size}, output(0), idx_in(0) {	}

	void setup_test() {
		v.reserve(n);
		for(long i=0; i<n; ++i) v.push_back(i);
	}

	void clear_test() {
		v.clear();
		cerr << "Output is: " << output << endl;
	}


	void run_multiple(const dynamic_execution & e, int cores) {
		grppi::pipeline(e,
				[&]() -> experimental::optional<long> {
			if (idx_in < v.size()) {
				idx_in++;
				return v[idx_in-1];
			} else
				return {};
		},
		grppi::farm(cores, // actually unused: it uses execution policy's conc_degree
				[&](long x) {
			long res = do_work(x); // basically waste time and do some math
			return res;
		}),
		[&](long x) {
			output += x; // atomically add the result
		}
		);
	}
};

int main(int argc, char **argv) {

	using namespace chrono;

	if(argc < 3) {
		cerr << "Usage: farm_GRPPI cores input_size" << endl;
		return -1;
	}

	int cores = stoi(argv[1]);
	int insize = stoi(argv[2]);

	parallel_execution_ff e(cores);

	FarmTest* test = new FarmTest(insize);
	test->setup_test();

	auto t1 = system_clock::now();
	test->run_multiple(e, cores);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	test->clear_test();
	delete test;

	cerr << "[FARM-GRPPI] Cores: " << cores
			<< "; InSize: " << insize
			<< "; Time: " << diff.count() << "ms"
			<< endl;
}
