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
#include <ppi/stream_reduce.h>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int reduce_example1(){
    /* Accumulation variable initialized in 0 and a vector of 0s with 10000 positions */
    int reduce_var = 0;
    std::vector<int> stream( 10000 );
    int index = 0;

    auto p = parallel_execution_thrust(1, thrust::cuda::par);

    stream_reduce( p,
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
        [] __device__ ( int v, int local_red ) {
                return local_red + v; // add 0 to each vector element 
        },
        // Reduce join as lambda
        []( int a , int & reduce_var) {
            reduce_var += a; // add all partial reductions in a accumulation variable
        }, reduce_var
    );

    return reduce_var; // return accumulation avariable
}

TEST(GrPPI, stream_reduce_gpu ){
    EXPECT_EQ(0, reduce_example1() );
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
