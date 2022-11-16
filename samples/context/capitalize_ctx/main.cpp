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
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>

// grppi
#include "grppi/pipeline.h"
#include "grppi/farm.h"
#include "grppi/context.h"
#include "grppi/common/execution_traits.h"

// Samples shared utilities
#include "../../util/util.h"

void capitalize(grppi::dynamic_execution & ex,
                std::istream & ifile, std::ostream & ofile)
{
  using namespace std;

  if constexpr (grppi::supports_context<decltype(ex)>()) {
  grppi::parallel_execution_native inner_ex{2};

  grppi::pipeline(ex,
    [&ifile]() -> grppi::optional<string> {
      string line;
      getline(ifile, line);
      if (!ifile) return {};
      return line;
    },
    run_with(ex, grppi::farm(4,
        [](const string & line) {
          string r;
          for (auto x : line) {
            r.push_back(std::toupper(x));
          }
          return r;
        }
      )
    ),
    [&ofile](const std::string & line) {
      ofile << line << "\n";
    }
  );
  }

}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " file_name outermode" << endl;
  cerr << "  file_name: Path to a plain text file" << endl;
  cerr << "  outermode:" << endl;
  print_available_modes(cerr);
}


int main(int argc, char **argv) {
    
  using namespace std;

  if(argc < 3){
    print_message(argv[0], "Invalid number of arguments.");
    return -1;
  }

  ifstream infile{argv[1]};
  if (!infile) {
    print_message(argv[0], "Cannot open file "s + argv[1]);
    return -1;
  }

  ofstream outfile{argv[2]};
  if (!outfile) {
    print_message(argv[0], "Cannot open file "s + argv[2]);
    return -1;
  }

  if (!run_test(argv[3], capitalize, infile, outfile)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
