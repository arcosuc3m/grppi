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
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <include/stream_filter.h>
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


int filter_example(auto &p) {
 
#ifndef NTHREADS
#define NTHREADS 6
#endif

    std::atomic<int> output;
    output = 0;
   
    std::ifstream is{"txt/file.txt"};
    if (!is.good()) { cerr << "TXT file not found!" << endl; return 0; }
    std::ofstream os{"txt/out.txt"};

    stream_filter(p,
        [&]() {
            auto v = read_list(is);
            return (v.size() == 0) ? optional<std::vector<int>>() : optional<std::vector<int>>(v); 
        },
        [&](const std::vector<int> v){
           //std::cout<<"FILTERING\n";
           output++;  //count number of filters done
           return (v.size()>10) ? true : false;
        }, 
        [&os](const std::vector<int> v ) {
           copy( begin(v) , end(v), ostream_iterator<int>{os," "});
           os << endl;
        }
    );

    return output;
}


TEST(GrPPI, filter_example_seq ){
    sequential_execution p{};
    EXPECT_EQ(1000, filter_example(p) );
}

TEST(GrPPI, filter_example_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(1000, filter_example(p) );
}

#ifdef OMP_ENABLE
    TEST(GrPPI, filter_example_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(1000, filter_example(p) );
    }
#endif
#ifdef TBB_ENABLE
    TEST(GrPPI, filter_example_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(1000, filter_example(p) );
    }
#endif


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

