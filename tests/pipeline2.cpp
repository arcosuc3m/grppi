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

#include <algorithm>
#include <pipeline.h>

using namespace std;
using namespace grppi;
template <typename T>
using optional = std::experimental::optional<T>;

void pipeline_example2() {

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

    std::vector<string> output;
    p.ordering=true;
    ifstream fe("txt/words.txt");
    if (!fe.good()) return;
    int numchar = 0;

    pipeline( p,
        // Pipeline stage 0
        [&]() -> optional<char>{
            char r; 
            fe >> r;
            if ( fe.eof() ) {
                return {}; 
            }
            else { 
		        //cout << r;
                return r;
            }
        },

        // Pipeline stage 1
        [&]( char k ) {
            std::string ss; 
            numchar++; 
            ss = k+std::to_string( numchar );
            return ss;
        },

        // Pipeline stage 2
        [&]( std::string l ) {
            //std::cout << l << std::endl;
            output.push_back(l);
        }
    );
    // Print results
//    std::sort(output.begin(), output.end());
    for (int i = 0; i < output.size(); i++){
        std::cout << output[i] << std::endl;
    }

}

int main() {

        //$ auto start = std::chrono::high_resolution_clock::now();
        pipeline_example2();
        //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

        //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
        //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

        return 0;
}
