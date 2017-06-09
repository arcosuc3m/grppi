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
#include <pipeline.h>
#include <stream_filter.h>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int pipeline_filter_example(auto &p, auto &f) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    int output = 0;
    int a = 10;
p.ordering=true;

    pipeline( p,
        // Pipeline stage 0
        [&]() { 
            a--; 
            //std::cout << "Stage 0\n";
            if (a == 0) 
                return optional<int>(); 
            else 
                return optional<int>(a); 
        },

        // Pipeline stage 1
        stream_filter(f, [&]( int k ) {
              if (k%2==0) {
                  //std::cout << "Discard " << k << "\n";
                  return false;
              } else {
                  //std::cout << "Accept " << k << "\n";
                  return true;
              }
        }),

        // Pipeline stage 2
        [&]( int k) {
            output += k;
        }
    );

    return output;
}
 


TEST(GrPPI, pipeline_filter_example_seq ){
    sequential_execution p{}, f{};
    EXPECT_EQ(25, pipeline_filter_example(p,f) );
}

TEST(GrPPI, pipeline_filter_example_thr ){
    parallel_execution_thr p{3}, f{NTHREADS-3};
    EXPECT_EQ(25, pipeline_filter_example(p,f) );
}

#ifdef GRPPI_OMP
    TEST(GrPPI, pipeline_filter_example_omp ){
        parallel_execution_omp p{3}, f{NTHREADS-3};
        EXPECT_EQ(25, pipeline_filter_example(p,f) );
    }
#endif
#ifdef GRPPI_TBB
    /* Not yet implemented */
    /*TEST(GrPPI, pipeline_filter_example_tbb ){
        parallel_execution_tbb p{3}, f{NTHREADS-3};
        EXPECT_EQ(25, pipeline_filter_example(p,f) );
    }*/
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

