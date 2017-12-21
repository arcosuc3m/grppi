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

#include "ff/node.hpp"
#include "ff/farm.hpp"
#include "ff/pipeline.hpp"

#include "supported_executions.h"

using namespace std;

long * Fworker(long *val, ff::ff_node*const ) {
	for(register int i=0;i<1000;++i){
		*val = *val + i;
	}
	return val;
}

struct generator : ff::ff_node {
	generator(long n) : size{n} { }
	void * svc(void *) {
		for(long i=0;i<size;++i) {
			long * task = new long( i );
			ff_send_out(task);
		}
		return EOS;
	}
	long size;
};

class Collector : public ff::ff_node {
	long output;
public:
	Collector() : output(0) {}
	void * svc(void *task) {
		long* t = (long*) task;
		output += *t;
		delete t;
		return GO_ON;
	}

	void svc_end() {
		cerr << "Output is: " << output << endl;
	}
};


class FarmTest {
	long n;
public:
	FarmTest(long size) :
		n{size} { }

		void run_multiple_ff(int cores) {
			ff::ff_Pipe<> pipe(make_unique<generator>(n),
					make_unique<ff::ff_OFarm<long>>(Fworker, cores),
					make_unique<Collector>()
			);

			pipe.run_and_wait_end();
		}
};


int main(int argc, char **argv) {

	using namespace chrono;

	if(argc < 3) {
		cerr << "Usage: farm_FF cores input_size" << endl;
		return -1;
	}

	int cores = stoi(argv[1]);
	int insize = stoi(argv[2]);

	FarmTest* test = new FarmTest(insize);

	auto t1 = system_clock::now();
	test->run_multiple_ff(cores);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	delete test;

	cerr << "[FARM-FF] Cores: " << cores
				<< "; InSize: " << insize
				<< "; Time: " << diff.count() << "ms"
				<< endl;
}
