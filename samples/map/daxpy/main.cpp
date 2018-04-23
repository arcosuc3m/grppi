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
#include <random>

// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void test_map(grppi::dynamic_execution & e, int n) {
  using namespace std;

  mt19937_64 rengine;
  uniform_real_distribution<double> value_gen{-100.0, 100.0};
  uniform_real_distribution<double> coef_gen{1.0, 10.0};

  vector<double> x;
  generate_n(back_inserter(x), n,
    [&]() { return value_gen(rengine); });

  vector<double> y;
  generate_n(back_inserter(y), n,
    [&]() { return value_gen(rengine); });
  double a = coef_gen(rengine);

  grppi::map(e, make_tuple(begin(x),begin(y)), end(x), begin(y),
    [a](int vx, int vy) { return a * vx + vy; });

  copy(begin(y), end(y), ostream_iterator<int>(cout, " "));
  cout << endl;
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
