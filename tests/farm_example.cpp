/**
* @version		GrPPI v0.2
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
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
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

#include <chrono>
#include <farm.h>

using namespace std;
using namespace grppi;

std::vector<int> read_list(std::istream & is){
  std::vector<int> result;
  string line;
  is >> ws;
  if(!getline(is,line)) return result;
  istringstream iline{line};
  int x;
  while(iline >> x){
    result.push_back(x);
  }
  return result;
}

std::string read_line(std::istream & is){
  std::string res;
  std::string line;
  is >> ws;
  if(!getline(is,line)) return res;
  return line;
}
 

void farm_example1() {

#ifndef NTHREADS
#define NTHREADS 6
#endif

#ifdef SEQ
    sequential_execution p{};
#elif OMP
    parallel_execution_omp p{NTHREADS};
#elif TBB
    parallel_execution_tbb p{NTHREADS};
#elif THR
    parallel_execution_native p{NTHREADS};
#else
    sequential_execution p{};
#endif

    std::ifstream is{"txt/filelist.txt"};
    if (!is.good()) { cerr << "TXT file not found!" << endl; return; }

    farm(p,
        // farm generator as lambda
        [&]() {
            auto f = read_line(is);
            
            return ( f.empty() ) ? optional<std::string>( ) : optional<std::string>( f );
        },

        // farm kernel as lambda
        [&]( std::string fname ) {
            std::fstream file (fname);
            auto v = read_list(file);
            int acumm = 0;
            for(int j = 0; j< v.size() ; j++) {
                acumm += v[j];
            }
            file << acumm;
        }
    );
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    farm_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
