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
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <pipeline.h>
#include <gtest/gtest.h>
#include <atomic>

using namespace std;
using namespace grppi;

std::vector<int> read_list(std::istream & is){
  std::vector<int> result;
  string line;
  is >> ws;
  if(!getline(is,line)) return result;
  istringstream iline{line};
  int x;
  while(iline >> x){
    result.push_back(x);
  }
  return result;
}

int pipeline_example(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    std::atomic<int> output;
    output = 0;

    ifstream is("txt/file.txt");
    if (!is.good()) { cerr << "TXT file not found!" << endl; return 0; }
    int numchar = 0;
    p.ordering=true;

    pipeline( p,
        // Pipeline stage 0
        [&]() {
             auto v = read_list(is);
             return ( v.size() == 0) ? optional<std::vector<int>>() : optional<std::vector<int>>(v);
        },

        // Pipeline stage 1
        [&]( const std::vector<int> & v) {
            return (v.size () > 0 ) ? *max_element(begin(v), end(v)) : numeric_limits<int>::min();

        },

        // Pipeline stage 2
        [&] ( int v ) {
            output += v;
        }
    );
    
    return output;
}



TEST(GrPPI, pipeline_example_seq ){
    sequential_execution p{};
    EXPECT_EQ(99402, pipeline_example(p) );
}

TEST(GrPPI, pipeline_example_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(99402, pipeline_example(p) );
}

#ifdef GRPPI_OMP
    TEST(GrPPI, pipeline_example_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(99402, pipeline_example(p) );
    }
#endif
#ifdef GRPPI_TBB
    TEST(GrPPI, pipeline_example_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(99402, pipeline_example(p) );
    }
#endif


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
