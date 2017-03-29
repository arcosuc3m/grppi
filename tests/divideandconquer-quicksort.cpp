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
#include "ppi/divideandconquer.hpp"
#include <random>

using namespace std;
using namespace grppi;

void dividec_example1() {

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

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0,5000000);

    std::vector<int> v(100000000);
    for( int i= 0 ; i< v.size(); i++){
        v[i] = uni(rng);
    }
    std::vector<int> out;
    
    DivideAndConquer(p,v, out,
                     [&](vector<int> & v){
        std::vector<std::vector<int>> subproblem;
        if(v.size() == 1){ subproblem.push_back(v);return subproblem; }

        std::vector<int> biggers;
        std::vector<int> smallers;


        int first = v[0];
        for(int i=1; i<v.size(); i++) {
            if (v[i] < first)
                smallers.push_back(v[i]); 
            else
                biggers.push_back(v[i]);  
        }

        if(smallers.size() > 0) subproblem.push_back(smallers);
        subproblem.push_back(vector<int> (1,first));
        if(biggers.size() > 0) subproblem.push_back(biggers);

        return subproblem;
        },
        [&](const vector<int> & problem, vector<int> & out){
            out.push_back(problem[0]);
              
        },
        [&](auto & partial, auto & out){
          for(int i = 0; i < partial.size(); i++)
              out.push_back(partial[i]);
        }
    );
}



int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    dividec_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
