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
  using iterator = std::vector<int>::iterator;
  iterator first, last;
  
  range() : first{}, last{} {}
  range(iterator f, iterator l) : first{f}, last{l} {}

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

  grppi::divide_conquer(exec,
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
