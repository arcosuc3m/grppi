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
#include <gtest/gtest.h>
#include "../ppi/enable_flags.hpp"

using namespace std;
using namespace grppi;

int fibonacci_example(auto &p) {

int output; // Output value

#ifndef NTHREADS
#define NTHREADS 6
#endif

    for(int v=0;v<40;v++){
      int find = 1;
      int out = 0;
      DivideAndConquer(p, v, out,
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
      
      output = out; 
   }

   return output;

}


TEST(GrPPI, divideandconquer_fibonacci_GT_seq ){
    sequential_execution p{};
    EXPECT_EQ(63245986, fibonacci_example(p) );
}

TEST(GrPPI, divideandconquer_fibonacci_GT_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(63245986, fibonacci_example(p) );
}

#ifdef OMP_ENABLE
    TEST(GrPPI, divideandconquer_fibonacci_GT_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(63245986, fibonacci_example(p) );
    }
#endif
#ifdef TBB_ENABLE
    TEST(GrPPI, divideandconquer_fibonacci_GT_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(63245986, fibonacci_example(p) );
    }
#endif

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
