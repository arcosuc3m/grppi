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
#include <gtest/gtest.h>
#include "enable_flags.hpp"

using namespace std;
using namespace grppi;

int pipeline_example1(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    int out = 0;
    int a = 10;
    std::vector<string> output;
    p.ordering=true;
    pipeline( p,
        // Pipeline stage 0
        [&]() { 
            a--; 
            if (a == 0) 
                return optional<int>(); 
            else 
                return optional<int>(a); 
        },

        // Pipeline stage 1
        [&]( int k ) {
            std::string ss; 
            ss = "t " + std::to_string( k );
            return std::string( ss );
        },

        // Pipeline stage 2
        [&]( std::string l ) {
            output.push_back("Stage 2 " + l);
        }
    );

    for (int i = 0; i < output.size(); i++){
        out++; // increase 1 for each task Stage 2 has finished
    }
    return out;
}
 
TEST(GrPPI, pipeline_example1_seq ){
    sequential_execution p{};
    EXPECT_EQ(9, pipeline_example1(p) );
}

TEST(GrPPI, pipeline_example1_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(9, pipeline_example1(p) );
}

#ifdef OMP_ENABLE
    TEST(GrPPI, pipeline_example1_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(9, pipeline_example1(p) );
    }
#endif
#ifdef TBB_ENABLE
    TEST(GrPPI, pipeline_example1_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(9, pipeline_example1(p) );
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
