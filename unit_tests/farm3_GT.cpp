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
#include <atomic>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <farm.h>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int farm_example3(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    int a = 20000;
    std::atomic<int> output;
    output = 0;

    farm(p,
        // Farm generator as lambda
        [&]() {
            a--; 
            if ( a == 0 ) 
                return optional<int>(); 
            else
                return optional<int>( a );
        },

        // Farm kernel as lambda
        [&]( int l ) {
            output += l*10;
        }
    );

    //std::cout << output << std::endl;
    return output;
}


TEST(GrPPI, farm3_seq ){
    sequential_execution p{};
    EXPECT_EQ(1999900000, farm_example3(p) );
}

TEST(GrPPI, farm3_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(1999900000, farm_example3(p) );
}

#ifdef OMP_ENABLE
    TEST(GrPPI, farm3_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(1999900000, farm_example3(p) );
    }
#endif
#ifdef TBB_ENABLE
    TEST(GrPPI, farm3_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(1999900000, farm_example3(p) );
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
