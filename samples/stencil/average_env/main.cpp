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
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void compute_avg(grppi::polymorphic_execution & e, int n) {
  using namespace std;

  vector<long long> in;
  generate_n(back_inserter(in), n,
    [i=0]() mutable { i++; return i*i; }); 

  vector<double> out(n);

  grppi::stencil(e, begin(in), end(in), begin(out),
    [](auto it, auto n) {
      return (*it + std::accumulate(begin(n), end(n), 0)) / double(n.size()+1);
    },
    [&in](auto it) {
      vector<double> r;
      if (it!=begin(in)) r.push_back(*prev(it));
      if (std::distance(it,end(in))>1) r.push_back(*next(it));
      return r;
    }
  );

  copy(begin(out), end(out), ostream_iterator<double>(cout, " "));
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

  if (!run_test(argv[2], compute_avg, n)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
