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
#include <farm.h>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int pipeline_farm_example(auto& p, auto& f) {

#ifndef NTHREADS
#define NTHREADS 6
#endif


    int out = 0;
    int n=10;
    std::vector<string> output;
p.ordering=true;

    pipeline(p,
             // Pipeline stage 0
             [&]() {
        std::vector<int> v(5);
        for ( int i = 0; i < 5; i++ )
            v[ i ] = i + n;

        if ( n < 0 )
            return optional< std::vector<int> >();
        n--;
        return optional<std::vector<int>>(v);
    },
    farm(f,
         [&](std::vector<int> v) {
        std::vector<long> acumm( v.size() );
        for(unsigned i = 0; i < acumm.size(); i++ ){
            acumm[i] = 0;
            for(auto j : v){
                acumm[i] += j;
            }
        }
        return (acumm);
    }
    ),
            // Pipeline stage 2
            [&]( std::vector<long> acc ) {
        double acumm = 0;
        for ( int i = 0; i < acc.size(); i++ )
            acumm += acc[ i ];

        return acumm;
    },

    // Pipeline stage 3
    [&]( double v ) {
        out += v;
    }
    );

    return out;
}



TEST(GrPPI, pipeline_farm_example_seq ){
    sequential_execution p{}, f{};
    EXPECT_EQ(1925, pipeline_farm_example(p,f) );
}

TEST(GrPPI, pipeline_farm_example_thr ){
    parallel_execution_thr p{3}, f{NTHREADS-3};
    EXPECT_EQ(1925, pipeline_farm_example(p,f) );
}

#ifdef OMP_ENABLE
    TEST(GrPPI, pipeline_farm_example_omp ){
        parallel_execution_omp p{3}, f{NTHREADS-3};
        EXPECT_EQ(1925, pipeline_farm_example(p,f) );
    }
#endif
#ifdef TBB_ENABLE
    TEST(GrPPI, pipeline_farm_example_tbb ){
        parallel_execution_tbb p{3}, f{NTHREADS-3};
        EXPECT_EQ(1925, pipeline_farm_example(p,f) );
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
