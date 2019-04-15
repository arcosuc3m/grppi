/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Standard library
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>

// grppi
#include "grppi/grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void test_map(grppi::dynamic_execution & e, int n) {
  using namespace std;
  using namespace chrono;

  vector<long> v;
  v.reserve(n);
  generate_n(back_inserter(v), static_cast<long>(n),
      [i=1]() mutable { return i++; });

  auto t1 = system_clock::now();
  
  auto r = grppi::reduce(e, begin(v), end(v), 0L,
    [](auto x, auto y) { return x+y; });

  auto t2 = system_clock::now();
  auto diff = duration_cast<milliseconds>(t2-t1);

  cout << "sum(0,...," << n << ")= " << r << endl;
  cout << "Reduction time: " << diff.count() << " ms" << endl;  
}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " size mode" << endl;
  cerr << "  size: Integer value with problem size" << endl;
  cerr << "  mode:" << endl;
  print_available_modes(cerr);
}


int main(int argc, char **argv) {
    
  using namespace std;

  if(argc < 3){
    print_message(argv[0], "Invalid number of arguments.");
    return -1;
  }

  int n = stoi(argv[1]);
  if(n <= 0){
    print_message(argv[0], "Invalid problem size. Use a positive number.");
    return -1;
  }

  if (!run_test(argv[2], test_map, n)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
