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
#include <include/reduce.h>
#include <functional>
#include <limits>
#include <gtest/gtest.h>
#include "../include/enable_flags.hpp"

using namespace std;
using namespace grppi;

double reduce_example1(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    std::vector<double> in(10);
    for(int i=0;i<in.size();i++) in[i] = (i+1);
    double out=1;
    reduce(p, in.begin(), in.end(), out, std::divides<double>());
    return out;
}

TEST(GrPPI, reduce_example1_seq ){
    sequential_execution p{};
    EXPECT_NEAR(3628799.9999999995, reduce_example1(p), 0.001 );
}

TEST(GrPPI, reduce_example1_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_NEAR(3628799.9999999995, reduce_example1(p), 0.001 );
}

#ifdef OMP_ENABLE
    TEST(GrPPI, reduce_example1_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_NEAR(3628799.9999999995, reduce_example1(p), 0.001 );
    }
#endif
#ifdef TBB_ENABLE
    TEST(GrPPI, reduce_example1_tbb ){
        double inf = std::numeric_limits<double>::infinity();
        parallel_execution_tbb p{NTHREADS};
        EXPECT_NEAR(3628799.9999999995, reduce_example1(p), 0.001);
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
