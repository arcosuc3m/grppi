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
#include <experimental/optional>

// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void test_map(grppi::dynamic_execution & e, int n, int window_size, int offset) {
  using namespace std;
  using namespace experimental;

  int i = 0;
  auto generator = [&i,n]() -> optional<int> {
    if (i<n) return i++;
    else     return {};
  };

  grppi::pipeline(e,
    generator,
    grppi::reduce(window_size, offset, 0,
      [](int x, int y) { return x+y; }),
    [](int x) { cout << x << endl; }
  );
}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " size window_size offset mode" << endl;
  cerr << "  size: Size of the initially generated sequence" << endl;
  cerr << "  window_size: Integer value with window size" << endl;
  cerr << "  offset: Integer value with offset" << endl;
  cerr << "  mode:" << endl;
  print_available_modes(cerr);
}


int main(int argc, char **argv) {
    
  using namespace std;

  if (argc < 5) {
    print_message(argv[0], "Invalid number of arguments.");
    return -1;
  }

  int size = stoi(argv[1]);
  if (size<=0) {
    print_message(argv[0], "Invalid sequence size. Use a positive number.");
    return -1;
  }

  int window_size = stoi(argv[2]);
  if (window_size<=0) {
    print_message(argv[0], "Invalid window size. Use a positive number.");
    return -1;
  }

  int offset = stoi(argv[3]);
  if (offset<=0) {
    print_message(argv[0], "Invalid offset. Use a positive number.");
    return -1;
  }

  if (!run_test(argv[4], test_map, size, window_size, offset)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
