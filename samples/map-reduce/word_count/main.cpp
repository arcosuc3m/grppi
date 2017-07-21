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
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>

// grppi
#include "mapreduce.h"

// Samples shared utilities
#include "../../util/util.h"

void test_mapreduce(grppi::polymorphic_execution & e,
                std::istream & file)
{
  using namespace std;

  vector<string> lines;
  for (string line; std::getline(file,line);) {
     lines.push_back(line);
  }
  map<string,int> init;

  auto result = map_reduce(e,
    lines.begin(),
    lines.end(),
    init,
    [](string & l){
      // Split lines in substrings representing words
      istringstream line{l};
      vector<string> words{istream_iterator<string>{line},
                           istream_iterator<string>{}};
      std::map<string,int> word_count;
      // Initialize map with the line words
      for (auto & w : words) { word_count[w]++; }
      return word_count;
    },
    [](auto partial_count, auto word_count){
      // Compute partial results
      for (auto & w : word_count) {
        partial_count[w.first]+= w.second;
      }
      return partial_count;
    }
  );

  std::cout << "Word : count " << std::endl;
  for (auto && w : result) {
    std::cout << w.first << " : " << w.second << std::endl;
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
