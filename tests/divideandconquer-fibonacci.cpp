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
#include "include/divideandconquer.h"

using namespace std;
using namespace grppi;

void fibonacci_example() {
/*    std::vector<std::vector<int>> mat(100);
    for(int i=0;i<mat.size();i++) {
        mat[i] = std::vector<int> (100);
        for(int j=0;j<mat[i].size();j++){
            mat[i][j] = 1;
        }
    }
*/
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

    std::cout<<"Fibonacci\n";
    for(int v=0;v<40;v++){
    int find = 1;
    int out = 0;
    divide_and_conquer(p, v, out,
        [&](auto & v){
           std::vector< int > subproblem;
    	   if(v<2) subproblem.push_back(v);
           else{
              subproblem.push_back(v-1);
               subproblem.push_back(v-2);
           }
           return subproblem;
        },
        [&](auto & problem, auto & partial){
           if(problem==0) partial = 0;
           else{
              int a=1, b=1;
              for(int i = 3; i <= problem; i++){
                 int c = a + b;
                 a = b;
                 b = c;
             }
             partial = b;
           }
        },
        [&](auto & partial, auto & out){
           out += partial;
        }
    );

    std::cout << v<< ":" << out << std::endl;
   }
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    fibonacci_example();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
