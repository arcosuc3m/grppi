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
#include <chrono>
#include <experimental/optional>

#include <pipeline.h>
#include <stream_filter.h>
using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

void pipeline_example1() {

#ifndef NTHREADS
#define NTHREADS 6
#endif

#ifdef SEQ
    sequential_execution p{}, f{};
#elif OMP
    parallel_execution_omp p{3}, f{NTHREADS-3};
#elif TBB
    #error TBB not yet implemented!
#elif THR
    parallel_execution_native p{3}, f{NTHREADS-3};
#else
    parallel_execution_ff p{3}, f{NTHREADS-3};
#endif

    int a = 10;
    p.enable_ordering();

    pipeline( p,
        // Pipeline stage 0
        [&]() -> optional<int>{ 
            a--; 
            //std::cout << "Stage 0\n";
            if (a == 0) 
                return {}; 
            else 
                return a; 
        },

        // Pipeline stage 1
        stream_filter(f, [&]( int k ) {
              if (k%2==0) {
                  //std::cout << "Discard " << k << "\n";
                  return false;
              } else {
                  //std::cout << "Accept " << k << "\n";
                  return true;
              }
        }),

        // Pipeline stage 2
        [&]( int k) {
            std::cout << "Sink: " << k << std::endl;
        }
    );
}
 
int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    pipeline_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
