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
#include <farm.h>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int farm_example2(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    std::vector<int> v(1000);
    std::vector<int> acumm(v.size());

    for( int i = 0; i < v.size(); i++ ) {
        v[i] = i;
    }

    int idx = 0;

    farm(p,
        // Farm generator as lambda
        [&]() { 
             if ( idx < v.size() ) {
                 idx++;
                 return optional<int>( (idx-1) );
           } else 
                 return optional<int>();
        },

        // Farm kernel as lambda
        [&]( int index ) {
             for ( int j = 0; j < v.size(); j++ ) {
                 acumm[index] += v[j];
             }
        }
    );

    // The output is the sum of the results of all threads
    int output = 0;
    for ( int i = 0; i < acumm.size(); i++ ) {
        output += acumm[i];
    }
    return output;
}


TEST(GrPPI, farm2_seq ){
    sequential_execution p{};
    EXPECT_EQ(499500000, farm_example2(p) );
}

TEST(GrPPI, farm2_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(499500000, farm_example2(p) );
}

#ifdef GRPPI_OMP
    TEST(GrPPI, farm2_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(499500000, farm_example2(p) );
    }
#endif
#ifdef GRPPI_TBB
    TEST(GrPPI, farm2_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(499500000, farm_example2(p) );
    }
#endif


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
