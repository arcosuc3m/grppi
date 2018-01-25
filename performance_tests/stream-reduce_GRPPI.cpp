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

#include "stream_reduce.h"
#include "pipeline.h"

#include "supported_executions.h"

using namespace std;
using namespace grppi;

class StreamReduceTest {
	long n;
	atomic<long> output;
	vector<long> v;
	int window_size;
	int offset;
	long idx_in;
public:

	StreamReduceTest(int size=1000) :
		n{size}, window_size{0}, offset{0}, idx_in(0) { }

	void setup_test(int ws=4, int os=2) {
		window_size = ws;
		offset = os;
		v.reserve(n);
		for(long i=0; i<n; ++i) v.push_back(i);
	}

	void clear_test() {
		v.clear();
		long out = output.load();
		cout << "Output is: " << out << endl;
	}

	void run_sr(const dynamic_execution & e) {
		int i=0;
		int s = n;
		auto generator = [&]() -> experimental::optional<long> {
			if (idx_in < v.size()) {
				idx_in++;
				return v[idx_in-1];
			} else
				return {};
		};

		grppi::pipeline(e,
				generator,
				grppi::reduce(window_size, offset, 0L,
						[](long x, long y) { return x+y; }
				),
				[&](long x) { output += x; }
		);
	}
};

int main(int argc, char **argv) {

	using namespace chrono;

	if(argc < 6) {
		cerr << "Usage: stream-reduce_GRPPI window offset cores input_size exec_mode [ordered=T]" << endl;
		cerr << "Available exec_mode are:" << endl;
		print_available_modes(cerr);
		cerr << "Set \'cores\' to 0 to use all available physical cores (" << get_phys_cores() << ")." << endl;
		cerr << "Set \'ordered\' to F to use unordered versions of stream-reduce." << endl;
		return -1;
	}

	bool ord = true;
	if(argc == 7)
		if(std::strcmp(argv[6], "T") != 0)
			ord = false;

	int win = stoi(argv[1]);
	int off = stoi(argv[2]);
	int cores = stoi(argv[3]) != 0 ? stoi(argv[3]) : get_phys_cores();
	int insize = stoi(argv[4]);
	dynamic_execution e(execution_mode(argv[5], cores, ord));

	if(!e.has_execution()) {
		cerr << "Exec_mode " << argv[3] << " not supported" << endl;
		return -1;
	}

	StreamReduceTest* test = new StreamReduceTest(insize);
	test->setup_test(win, off);

	auto t1 = system_clock::now();
	test->run_sr(e);
	auto t2 = system_clock::now();
	auto diff = duration_cast<milliseconds>(t2-t1);

	test->clear_test();
	delete test;

	cout << "[STR_RED-GRPPI] Cores: " << cores
			<< "; InSize: " << insize
			<< "; WinSize: " << win
			<< "; Offset: " << off
			<< "; Time: " << diff.count() << "ms"
			<< endl;
}

