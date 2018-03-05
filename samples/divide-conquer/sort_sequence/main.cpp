/**
* @version    GrPPI v0.3
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
#include <iterator>
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

struct range {
  std::vector<int>::iterator first, last;
  
  auto size() const { return distance(first,last); }
  friend std::ostream & operator<<(std::ostream & os, const range & r);
};

std::ostream & operator<<(std::ostream & os, const range & r) {
  os << "(size= " << r.size() << ") ";
  std::copy(r.first, r.last, std::ostream_iterator<int>(std::cerr, " "));
  return os;
}

std::vector<range> divide(range r) {
  auto mid = r.first + distance(r.first,r.last)/2;
  return { {r.first,mid} , {mid, r.last} };
}

void sort_sequence(grppi::dynamic_execution & exec, int n) {
  using namespace std;

  std::random_device rdev;
  std::uniform_int_distribution<> gen{1,1000};

  vector<int> v;
  for (int i=0; i<n; ++i) {
    v.push_back(gen(rdev));
  }
  
  range problem{begin(v), end(v)};

  auto res = grppi::divide_conquer(exec,
    problem,
    [](auto r) -> vector<range> { return divide(r); },
    [](auto r) { return 1>=r.size(); },
    [](auto x) { return x; },
    [](auto r1, auto r2) {
      std::inplace_merge(r1.first, r1.last, r2.last);
      return range{r1.first, r2.last};
    }
  );

  copy(begin(v), end(v), ostream_iterator<int>(cout, " "));
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

  if (!run_test(argv[2], sort_sequence, n)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
