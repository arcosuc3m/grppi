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
#include "grppi/grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void discard_words(grppi::dynamic_execution & ex,
                  std::istream & file)
{
  using namespace std;
  using namespace experimental;

  grppi::pipeline(ex,
    [&file]() -> optional<string> {
      string word;
      file >> word;
      if (!file) { return {}; }
      else { return word; }
    },
    grppi::keep([](string w) -> bool { return w.length() < 4; }),
    [](string w) {
      cout << w << endl;
    }
  );

}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " file_name mode" << endl;
  cerr << "  file_name: Path to a plain text file" << endl;
  cerr << "  mode:" << endl;
  print_available_modes(cerr);
}


int main(int argc, char **argv) {
    
  using namespace std;

  if(argc < 3){
    print_message(argv[0], "Invalid number of arguments.");
    return -1;
  }

  ifstream file{argv[1]};
  if (!file) {
    print_message(argv[0], "Cannot open file "s + argv[1]);
    return -1;
  }

  if (!run_test(argv[2], discard_words, file)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
