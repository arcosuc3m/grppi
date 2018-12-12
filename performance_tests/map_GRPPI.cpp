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
#include <cstring>

#include "grppi.h"
#include "supported_executions.h"

using namespace std;
using namespace grppi;

class MapTest {
	// Vectors
	vector<double> v;
	vector<double> v2;
	vector<double> w;
	long n;
	double a=1;

public:

	MapTest(long size=1000) : n{size}, a(3.7573) {

	}

	void setup_daxpy() {
		v.reserve(n); 
		w = vector<double>(n);
		for(long i=0; i<n; ++i) {
			v.push_back( static_cast<double>(i*3.14) );
			v2.push_back( static_cast<double>((2*v[i])/1.7) );
		}
	}

	void clear_all() {
		v.clear(); v2.clear();
		auto sum = std::accumulate(w.begin(), w.end(), 0.0);
		w.clear();
		cerr << "Final sum is: " << sum << endl;
	}

	void run_daxpy(const dynamic_execution & exec) {
		grppi::map(exec,
				std::make_tuple(v.begin(),v2.begin()), v.end(), w.begin(),
		        [this](double & x, double & y)  {
     	 	    	return a*(x+y);
		        }
		);
	}
};

int main(int argc, char **argv) {

	using namespace chrono;

	if(argc < 4) {
		cerr << "Usage: map_GRPPI cores input_size exec_mode" << endl;
		cerr << "Available exec_mode are:" << endl;
		print_available_modes(cerr);
		cerr << "Set \'cores\' to 0 to use all available physical cores (" << get_phys_cores() << ")." << endl;
		return -1;
	}

	int cores = stoi(argv[1]) != 0 ? stoi(argv[1]) : get_phys_cores();
	long insize = stol(argv[2]);
	dynamic_execution e(execution_mode(argv[3], cores));

	if(!e.has_execution()) {
		cerr << "Exec_mode " << argv[3] << " not supported" << endl;
		return -1;
	}
	
	MapTest test(insize);
	test.setup_daxpy();

	auto t1 = system_clock::now();
	test.run_daxpy(e);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	test.clear_all();

	cerr << "[MAP-GRPPI] Cores: " << cores
			<< "; InSize: " << insize
			<< "; Time: " << diff.count() << "ms"
			<< endl;
}
