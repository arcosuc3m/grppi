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
#include <farm.h>

using namespace std;
using namespace grppi;

void farm_example1() {

#ifndef NTHREADS
#define NTHREADS 4
#endif

#if THR
    parallel_execution_thr p{NTHREADS};
#else
   #error Affinity is only supported for C++ thread implementation in the current version
#endif

    
    int a = 200000;
    //Thread binding
    for ( int i =0;i<NTHREADS; i++){
       //Bind each thread to the CPU core with the same number as the Thread ID.
       p.set_thread_affinity(i, {i});
    }
    //Numa binding
    for(int i = 0; i< NTHREADS; i++){
       //Bind thread alternatively to the numa nodes 1 and 2. If there only one numa node, this has no effects.
       p.set_numa_affinity(i, {i%2});
    }
    
    farm(p,
        // farm generator as lambda
        [&]() {
            //Thread 0 - Core 0 - NUMA 0
            a--; 
            if ( a == 0 ) 
                return optional<int>(); 
            else
                return optional<int>( a );
        },

        // farm kernel as lambda
        [&]( int l ) {
            //Thread 1 - Core 1 - NUMA 1
            //Thread 2 - Core 2 - NUMA 0
            //Thread 3 - Core 3 - NUMA 1
           //The memory allocation is done in both numa nodes and only cores 0-4 are active.
           std::vector<double> memtest(l,1);
           for(int i=0;i<memtest.size();i++) { memtest[i] += 1; }
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
