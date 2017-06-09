/**
* @version		GrPPI v0.1
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
#include <chrono>
#include <stream_reduce.h>
#include <string>
#include <sstream>
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



void stream_reduce_example() {
 
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
    parallel_execution_thr p{NTHREADS};
#else
    sequential_execution p{};
#endif

    ifstream is("txt/file.txt");
    if (!is.good()) { cerr << "TXT file not found!" << endl; return; }

    int reduce_var = 0;
    
    stream_reduce( p,
        // GenFunc: stream consumer
        [&]() {
            auto r = read_list(is);
            return ( r.size() == 0 ) ? optional<vector<int>>{} : optional<vector<int>>(r);
        },  	
        // TaskFunc: reduce kernel 
        [&]( vector<int> v ) {
            int loc_red = 0;
            for( int i = 0; i < v.size(); i++ )
                loc_red += v[i];
            return loc_red;
        },
        // RedFunc: final reduce
        [&]( int loc_red ,int &reduce_var ) {
            reduce_var += loc_red;
    }, reduce_var
);

    std::cout<<"Result: " << reduce_var << std::endl;
}


int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    stream_reduce_example();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}


