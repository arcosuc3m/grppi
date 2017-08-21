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
#include <random>

// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"

template <typename T>
class row_span {
public:
  row_span(std::vector<T> & v, int cols, int r) : vec_{v}, cols_{cols}, row_{r} {}
  T operator[](int i) const { return vec_[row_*cols_+i]; }

private:
  std::vector<T> & vec_;
  int cols_;
  int row_;
};

template <typename T>
class col_span {
public:
  col_span(std::vector<T> & v, int cols, int c) : vec_{v}, cols_{cols}, col_{c} {}
  T operator[](int i) const { return vec_[i*cols_+col_]; }

private:
  std::vector<T> & vec_;
  int cols_;
  int col_;
};

template <typename C, typename I>
int row_index(I it, C & cont, int ncols) {
  return std::distance(cont.begin(), it) / ncols;
}

template <typename C, typename I>
int col_index(I it, C & cont, int ncols) {
  return std::distance(cont.begin(), it) % ncols;
}

void matrix_mult(grppi::polymorphic_execution & e, int n) {
  using namespace std;

  random_device rdev;
  uniform_real_distribution<double> gen{1.0, 100.00};

  std::vector<double> a;
  generate_n(back_inserter(a), n*n,
    [&]() { return gen(rdev); });
  std::vector<double> b;
  generate_n(back_inserter(b), n*n,
    [&]() { return gen(rdev); });
  std::vector<double> c(n*n);

  grppi::stencil(e, begin(a), end(a), begin(c),
    [=](auto it, auto nh) {
      double r = 0;
      for (int k=0;k<n;++k) { r+= nh.first[k] * nh.second[k]; }
      return r;
    },
    [&](auto it1, auto it2) {
      return make_pair(
        row_span<double>{a, n, row_index(it1,a,n)},
        col_span<double>{b, n, col_index(it2,b,n)}
      );
    },
    begin(b)
  );

  cout << "size(a)" << a.size() << endl;
  copy(begin(a), end(a), ostream_iterator<double>(cout, " "));
  cout << endl << endl;
  cout << "size(b)" << b.size() << endl;
  copy(begin(b), end(b), ostream_iterator<double>(cout, " "));
  cout << endl << endl;
  cout << "size(c)" << c.size() << endl;
  copy(begin(c), end(c), ostream_iterator<double>(cout, " "));
  cout << endl << endl;
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

  if (!run_test(argv[2], matrix_mult, n)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
