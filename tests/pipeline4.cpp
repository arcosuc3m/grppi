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
#include <pipeline.h>

using namespace std;
using namespace grppi;

void pipeline_example1() {

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

    int a = 10;
    p.ordering=true;
    /* Test exit inmediately in the first stage */
    pipeline( p,
        // Pipeline stage 0
        [&]() { 
            return optional<int>(); 
        },

        // Pipeline stage 1
        [&]( int k ) {
            std::cout << "Stage 1\n";
            std::string ss; 
            ss = "t " + std::to_string( k );
            return std::string( ss );
        },

        // Pipeline stage 2
        [&]( std::string l ) {
            std::cout << "Stage 2 ";
            std::cout << l << std::endl;
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
