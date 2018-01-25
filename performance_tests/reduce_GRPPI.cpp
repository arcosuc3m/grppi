#include <iostream>
#include <streambuf>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>
#include <random>

#include "reduce.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;
 
class ReduceTest {

	// Vectors
	vector<double> v;
	long n;
	double a;
public:

	ReduceTest(long size=1000) : n{size}, a(3.7573) {

	}

	void setup_reduce() {
		v.reserve(n);
		for(long i=0; i<n; ++i) {
			v.push_back( static_cast<double>(i*3.14) );
		}
	}

	void clear_all() {
		v.clear();
	}

	auto run_sum(const dynamic_execution & e) {
		auto redend = grppi::reduce(e, begin(v), end(v), 0.0,
				[&](auto x, auto y) {
			return x+y;
		});
		return redend;
	}
};


int main(int argc, char **argv) {

	using namespace chrono;

	if(argc < 4) {
		cerr << "Usage: reduce_GRPPI cores input_size exec_mode" << endl;
		cerr << "Available exec_mode are:" << endl;
		print_available_modes(cerr);
		cerr << "Set \'cores\' to 0 to use all available physical cores (" << get_phys_cores() << ")." << endl;
		return -1;
	}

	int cores = stoi(argv[1]) != 0 ? stoi(argv[1]) : get_phys_cores();
	int insize = stoi(argv[2]);
	dynamic_execution e(execution_mode(argv[3], cores));

	if(!e.has_execution()) {
		cerr << "Exec_mode " << argv[3] << " not supported" << endl;
		return -1;
	}

	ReduceTest* test = new ReduceTest(insize);
	test->setup_reduce();

	auto t1 = system_clock::now();
	auto res = test->run_sum(e);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	test->clear_all();
	delete test;

	cerr << "Result is: " << res << endl;

	cerr << "[RED-GRPPI] Cores: " << cores
				<< "; InSize: " << insize
				<< "; Time: " << diff.count() << "ms"
				<< endl;
}
