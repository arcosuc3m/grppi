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
#include <atomic>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <include/farm.h>

using namespace std;
using namespace grppi;

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
    parallel_execution_thr p{NTHREADS};
#else
    sequential_execution p{};
#endif

    int a = 20000;
    std::atomic<int> output;
    output = 0;

    farm(p,
        // farm generator as lambda
        [&]() {
            a--; 
            if ( a == 0 ) 
                return optional<int>(); 
            else
                return optional<int>( a );
        },

        // farm kernel as lambda
        [&]( int l ) {
            output += l*10;
        }
    );

    std::cout << output << std::endl;
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    farm_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
