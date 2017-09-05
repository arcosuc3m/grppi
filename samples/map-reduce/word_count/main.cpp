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
#include <map>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>

// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void test_mapreduce(grppi::dynamic_execution & ex,
                    std::istream & file)
{
  using namespace std;

  vector<string> words;
  copy(istream_iterator<string>{file}, istream_iterator<string>{},
    back_inserter(words));

  // Word count for vector of lines
  auto result = map_reduce(ex,
    words.begin(), words.end(), map<string,int>{},
    [](string word) -> map<string,int> { return {{word,1}}; },
    [](map<string,int> & lhs, const map<string,int> & rhs) -> map<string,int> & {
      for (auto & w : rhs) {
        lhs[w.first]+= w.second;
      }
      return lhs;
    }
  );

  cout << "Word : count " << endl;
  for (const auto & w : result) {
    cout << w.first << " : " << w.second << endl;
  }
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

  if (!run_test(argv[2], test_mapreduce, file)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
