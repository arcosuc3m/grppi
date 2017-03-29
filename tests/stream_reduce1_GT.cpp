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
#include <ppi/stream_reduce.hpp>
#include <gtest/gtest.h>
#include "../ppi/enable_flags.hpp"

using namespace std;
using namespace grppi;

int stream_reduce_example1(auto& p){

#ifndef NTHREADS
#define NTHREADS 6
#endif

    int reduce_var=0;
    std::vector<int> stream( 10000, 1 );
    int index = 0;

    StreamReduce( p,
        // Reduce generator as lambda
        [&]() { 
            int n = ( stream.size() - index ) < 100 ? (stream.size() - index) : 100;
            if( n <= 0 ) 
                 return optional<std::vector<int>> ();

            std::vector<int> v(n);
            for ( int i = 0; i < n; i++ ) {
                 v[ i ] = stream[ index + i ];
            } 
            index += 100;
            return optional<std::vector<int>> ( v );
        },
        // Reduce kernel as lambda
        []( std::vector<int> v ) {
            int local_red = 0;
            for( int i = 0; i < v.size(); i++ ) 
                local_red += v[ i ];
            return local_red;
        },
        // Reduce join as lambda
        []( int a , int &reduce_var) {
            reduce_var += a;
        }, 
        reduce_var
    );

    return reduce_var;
}

TEST(GrPPI, stream_reduce_example1_seq ){
    sequential_execution p{};
    EXPECT_EQ(10000, stream_reduce_example1(p) );
}

TEST(GrPPI, stream_reduce_example1_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(10000, stream_reduce_example1(p) );
}

#ifdef OMP_ENABLE
    TEST(GrPPI, stream_reduce_example1_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(10000, stream_reduce_example1(p) );
    }
#endif
#ifdef TBB_ENABLE
    TEST(GrPPI, stream_reduce_example1_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(10000, stream_reduce_example1(p) );
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

