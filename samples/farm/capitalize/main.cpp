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
#include <cctype>

// grppi
#include "grppi/grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void capitalize(grppi::dynamic_execution & e, 
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
