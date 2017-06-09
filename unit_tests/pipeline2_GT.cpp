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
#include <algorithm>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int pipeline_example2(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    int out = 0;
    std::vector<string> output;
    p.ordering=true;
    ifstream fe("txt/words.txt");
    if (!fe.good()) return 0;
    int numchar = 0;

    pipeline( p,
        // Pipeline stage 0
        [&]() {
            char r; 
            if ( fe.eof() ) {
                return optional<char>(); 
            }
            else { 
                fe >> r;
		        //cout << r;
                return optional<char>(r);
            }
        },

        // Pipeline stage 1
        [&]( char k ) {
            std::string ss; 
            numchar++; 
            ss = k+std::to_string( numchar );
            return ss;
        },

        // Pipeline stage 2
        [&]( std::string l ) {
            //std::cout << l << std::endl;
            output.push_back(l);
        }
    );

    for (int i = 0; i < output.size(); i++){
        out++; // increase 1 for each task Stage 2 has finished
    }
    return out;
}


TEST(GrPPI, pipeline_example2_seq ){
    sequential_execution p{};
    EXPECT_EQ(378, pipeline_example2(p) );
}

TEST(GrPPI, pipeline_example2_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(378, pipeline_example2(p) );
}

#ifdef GRPPI_OMP
    TEST(GrPPI, pipeline_example2_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(378, pipeline_example2(p) );
    }
#endif
#ifdef GRPPI_TBB
    TEST(GrPPI, pipeline_example2_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(378, pipeline_example2(p) );
    }
#endif


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
