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
#include <chrono>
#include <reduce.h>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int reduce_example2(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif 

    std::vector<std::vector<int>> in(100);
    for(int i=0;i<in.size();i++) {
       in[i] = std::vector<int> (100);
       for(int k=0; k<in[i].size(); k++){
          in[i][k]= 1;
       }
    }
    std::vector<int> out(100);


    int out2 =0 ;
//    Reduce(in.begin(), in.end(), out2, [&](auto & in, int & out){  out += in; }  );
//    Reduce(in.begin(), in.end(), out2, [&](auto & in, auto & out){
//       Reduce(in.begin(), in.end(), out, [&](auto & in, auto & out){ out += in; });
//     },  [&](auto & in, auto & out){ out += in; } );
//

     reduce(p, in.begin(), in.end(), out.begin(), [&](auto & in, auto & out){
             reduce(p, in.begin(), in.end(), out,  std::plus<int>() );
         } 
     );

    reduce(p, out.begin(), out.end(), out2,  std::plus<int>() );
 
    return out2;
}

TEST(GrPPI, reduce_example2_seq ){
    sequential_execution p{};
    EXPECT_EQ(10000, reduce_example2(p) );
}

TEST(GrPPI, reduce_example2_thr ){
    parallel_execution_native p{NTHREADS};
    EXPECT_EQ(10000, reduce_example2(p) );
}

#ifdef GRPPI_OMP
    TEST(GrPPI, reduce_example2_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(10000, reduce_example2(p) );
    }
#endif
#ifdef GRPPI_TBB
    TEST(GrPPI, reduce_example2_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(10000, reduce_example2(p) );
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
