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
#include <cctype>

// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void capitalize(grppi::polymorphic_execution & e, 
                std::istream & in, std::ostream & out)
{
  using namespace std;
  using namespace experimental;

  grppi::pipeline(e,
    [&]() -> optional<string> {
      string word;
      in >> word;
      if (in) return word;
      else return {};
    },
    grppi::farm(4,
      [](auto word) { 
        grppi::sequential_execution seq{};
        grppi::map(seq, begin(word), end(word), begin(word), [](char c) { 
          return std::toupper(c);
        }); 
        return word;
      }),
    [&](auto word) {
      out << word << std::endl;
    }
  );
}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " mode input output" << endl;
  cerr << "  input: Input file name" << endl;
  cerr << "  output: Output file name" << endl;
  cerr << "  mode:" << endl;
  print_available_modes(cerr);
}


int main(int argc, char **argv) {
    
  using namespace std;

  if(argc < 4){
    print_message(argv[0], "Invalid number of arguments.");
    return -1;
  }

  ifstream in{argv[2]};
  if (!in) {
    print_message(argv[0], "Cannot open file "s + argv[2]);
    return -2;
  }

  ofstream out{argv[3]};
  if (!out) {
    print_message(argv[0], "Cannot open file "s + argv[3]);
    return -2;
  }

  if (!run_test(argv[1], capitalize, in, out)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
