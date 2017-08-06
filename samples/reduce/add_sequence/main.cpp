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

// grppi
#include "reduce.h"

// Samples shared utilities
#include "../../util/util.h"

void test_map(grppi::polymorphic_execution & e, int n) {
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
