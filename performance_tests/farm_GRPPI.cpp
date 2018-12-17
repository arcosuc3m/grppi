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
#include <cstring>
#include <experimental/optional>

#include "grppi.h"
#include "supported_executions.h"

using namespace std;
using namespace grppi;


class FarmTest {
	long do_work(long & val) {
	  for(int i=0;i<10000000;++i){
	    val += i;
        asm("");
	  }
	  return val;
	}

public:

	FarmTest(long size) : n{size}, output(0), idx_in(0) {	}

	void run_multiple(const dynamic_execution &e, int replicas) {
		grppi::pipeline(e,
			[&]() -> experimental::optional<long> {
			if (idx_in < n) {
				idx_in++;
				return idx_in;
			} else
				return {};
		},
		grppi::farm(replicas, // actually unused: it uses execution policy's conc_degree
		    [&](long x) {
			  long res = do_work(x); // basically waste time and do some math
        	  return res;
		}),
		[&](long x) {
			output += x; // atomically add the result
		}
		);
        std::cout<<output<<std::endl;
	}
private:
	long n;
	long output;
	long idx_in;
};

int main(int argc, char **argv) {

	using namespace chrono;
	if(argc < 4) {
		cerr << "Usage: farm_GRPPI cores input_size exec_mode [ordered=T]" << endl;
		cerr << "Available exec_mode are:" << endl;
		print_available_modes(cerr);
		cerr << "Set \'cores\' to 0 to use all available physical cores (" << get_phys_cores() << ")." << endl;
		cerr << "Set \'ordered\' to F to use unordered versions of farm." << endl;
		return -1;
	}

	bool ord = true;
	if(argc == 5)
		if(std::strcmp(argv[4], "T") != 0)
			ord = false;

	int cores = stoi(argv[1]) != 0 ? stoi(argv[1]) : get_phys_cores();
	int insize = stoi(argv[2]);
    auto e = execution_mode(argv[3], cores, ord);

	if(!e.has_execution()) {
		cerr << "Exec_mode " << argv[3] << " not supported" << endl;
		return -1;
	}

	FarmTest test(insize);
	auto t1 = system_clock::now();
	test.run_multiple(e, cores);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	cerr << "[FARM-GRPPI] Cores: " << cores
			<< "; InSize: " << insize
			<< "; Time: " << diff.count() << "ms"
			<< endl;

}
