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
#include <stream_reduce.h>

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

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
    parallel_execution_native p(NTHREADS);
#else
    sequential_execution p{};
#endif
    int total = 0;
    int reduce_var=0;
    std::vector<int> stream( 10000, 1 );
    int index = 0;
    int n=0;
    stream_reduce( p,
        //Window size
        1000000,
        1000000,
        0,
        // Reduce generator as lambda
        [&]() -> optional<int>{ 
            n++;
            if(n != 1000000000) 
              return  1;
            else
              return {};
        },
        // Reduce kernel as lambda
        std::plus<int>(),
        // Reduce join as lambda
        [&]( int a) {
            total += a;
            std::cout<<"PARTIAL REDUCE : "<<a<<" TOTAL " <<total<< std::endl;
        }
    );
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    reduce_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}


