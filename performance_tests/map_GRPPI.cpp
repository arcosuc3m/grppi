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

#include "map.h"
#include "supported_executions.h"

using namespace std;
using namespace grppi;

class MapTest {
	// Vectors
	vector<double> v;
	vector<double> v2;
	vector<double> w;
	long n;
	double a;

public:

	MapTest(long size=1000) : n{size}, a(3.7573) {

	}

	void setup_daxpy() {
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
		w.clear();
		cerr << "Final sum is: " << sum << endl;
	}

	void run_daxpy(const dynamic_execution & exec) {
		grppi::map(exec,
				v.begin(), v.end(), w.begin(),
				[this](double x, double y) {
			return a*(x+y);
		},
		v2.begin()
		);
	}
};

int main(int argc, char **argv) {

	using namespace chrono;

	if(argc < 3) {
		cerr << "Usage: map_GRPPI cores input_size" << endl;
		return -1;
	}

	int cores = stoi(argv[1]);
	int insize = stoi(argv[2]);

	parallel_execution_ff e(cores);

	MapTest* test = new MapTest(insize);
	test->setup_daxpy();

	auto t1 = system_clock::now();
	test->run_daxpy(e);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	test->clear_all();
	delete test;

	cerr << "[MAP-GRPPI] Cores: " << cores
				<< "; InSize: " << insize
				<< "; Time: " << diff.count() << "ms"
				<< endl;
}
