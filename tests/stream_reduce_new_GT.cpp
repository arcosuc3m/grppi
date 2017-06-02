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
#include <stream_reduce.h>
#include <gtest/gtest.h>
#include "enable_flags.hpp"

using namespace std;
using namespace grppi;

int reduce_example3(auto& p){

#ifndef NTHREADS
#define NTHREADS 6
#endif

    int total = 0;
    int reduce_var=0;
    std::vector<int> stream( 10000, 1 );
    int index = 0;
    int n=0;
    stream_reduce( p,
        // Reduce generator as lambda
        [&]() { 
            n++;
            if(n != 1000000000) 
              return (optional<int> ( 1 ));
            else
              return (optional<int> ());
        },
        //Window size
        1000000,
        1000000,
        // Reduce kernel as lambda
        std::plus<int>(),
        // Reduce join as lambda
        [&]( int a) {
            total += a;
        } 
    );
    return total;
}

TEST(GrPPI, reduce_example3_seq ){
    sequential_execution p{};
    EXPECT_EQ(999999999, reduce_example3(p) );
}

TEST(GrPPI, reduce_example3_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(999999999, reduce_example3(p) );
}

#ifdef OMP_ENABLE
    TEST(GrPPI, reduce_example3_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(999999999, reduce_example3(p) );
    }
#endif
#ifdef TBB_ENABLE
    TEST(GrPPI, reduce_example3_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(999999999, reduce_example3(p) );
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}



