/**
* @version    GrPPI v0.1
* @copyright    Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license    GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
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
#include "common/polymorphic_execution.h"
#include "stream_reduce.h"

// Samples shared utilities
#include "../../util/util.h"

void test_map(grppi::polymorphic_execution & e, int n, int window_size, int offset) {
  using namespace std;
  using namespace experimental;

  int i = 0;
  auto generator = [&i,n]() -> optional<int> {
    if (i<n) return i++;
    else     return {};
  };

  grppi::stream_reduce(e,
    generator, window_size, offset,
    [](int x, int y) { return x+y; },
    [](int x) { cout << x << endl; },
    0
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
    print_message(argv[0], "Invalid squence size. Use a positive number.");
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
