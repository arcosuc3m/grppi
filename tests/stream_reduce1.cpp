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
#include <ppi/stream_reduce.hpp>

using namespace std;
using namespace grppi;

void reduce_example1(){

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

    int reduce_var=0;
    std::vector<int> stream( 10000, 1 );
    int index = 0;

    StreamReduce( p,
        // Reduce generator as lambda
        [&]() { 
            int n = ( stream.size() - index ) < 100 ? (stream.size() - index) : 100;
            if( n <= 0 ) 
                 return optional<std::vector<int>> ();

            std::vector<int> v(n);
            for ( int i = 0; i < n; i++ ) {
                 v[ i ] = stream[ index + i ];
            } 
            index += 100;
            return optional<std::vector<int>> ( v );
        },
        // Reduce kernel as lambda
        []( std::vector<int> v ) {
            int local_red = 0;
            for( int i = 0; i < v.size(); i++ ) 
                local_red += v[ i ];
            return local_red;
        },
        // Reduce join as lambda
        []( int a , int &reduce_var) {
            reduce_var += a;
        }, 
        reduce_var
    );
    std::cout<<"OUTPUT : " << reduce_var << std::endl;
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    reduce_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}


