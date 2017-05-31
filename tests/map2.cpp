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
#include <include/map.h>

using namespace std;
using namespace grppi;

void map_example1() {

#ifndef NTHREADS
#define NTHREADS 6
#endif

#ifdef SEQ
    sequential_execution p{};
#elif OMP
    parallel_execution_omp p{};
#elif TBB
    #error Not yet implemented!
#elif THR
    parallel_execution_thr p{NTHREADS};
#else
    sequential_execution p{};
#endif

    std::vector<int> in(100);
    for(int i=0;i<in.size();i++) in[i] = i;
    std::vector<int> out(100);
    int i = 0;

    grppi::map( p,
        [&](){
            
             if( i < in.size()) {
                  optional<int> element(i);
                  i++;
                  return element;
             }else{
               return optional<int>();
            }
        },
        [&](int index){
             std::cout<<"COMPUTE"<<std::endl;
             out[index] = in[index];
        }
    );
		
		for(int i=0;i<in.size();i++){
			std::cout << out[i] << std::endl;
		}
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    map_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
