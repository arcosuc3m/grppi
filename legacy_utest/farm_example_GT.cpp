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
#include <farm.h>

#include <gtest/gtest.h>
#include <atomic>

using namespace std;
using namespace grppi;

std::vector<int> read_list(std::istream & is){
  std::vector<int> result;
  string line;
  is >> ws;
  if(!getline(is,line)){
   return result;
  }
  
  istringstream iline{line};
  int x;
  while(iline >> x){
    result.push_back(x);
  }
  return result;
}

std::string read_line(std::istream & is){
  std::string res;
  std::string line;
  is >> ws;
  if(!getline(is,line)) return res;
  return line;
}
 

int farm_example1(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    std::atomic<int> output;
    output = 0;

    std::ifstream is{"txt/filelist.txt"};
    if (!is.good()) { cerr << "TXT file not found!" << endl; return 0; } // 0 if error

    farm(p,
        // Farm generator as lambda
        [&]() {
            auto f = read_line(is);
            return ( f.empty() ) ? optional<std::string>( ) : optional<std::string>( f ) ;
        },

        // Farm kernel as lambda
        [&]( std::string fname ) {
            std::fstream file ("txt/"+fname);
            auto v = read_list(file);
            int acumm = 0;
            for(int j = 0; j< v.size() ; j++) {
                acumm += v[j];
            }
            file << acumm;
            output += acumm;
        }
    );

    return output;
}


TEST(GrPPI, farm_example_seq ){
    sequential_execution p{};
    EXPECT_EQ(505994, farm_example1(p) );
}

TEST(GrPPI, farm_example_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(505994, farm_example1(p) );
}

#ifdef GRPPI_OMP
    TEST(GrPPI, farm_example_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(505994, farm_example1(p) );
    }
#endif
#ifdef GRPPI_TBB
    TEST(GrPPI, farm_example_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(505994, farm_example1(p) );
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
