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
#include <stencil.h>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int stencil_example1(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

  std::vector<int> out(100);
  int output=0;

   std::vector<int> in(100);
   for(int i=0;i<in.size();i++) in[i] = i;
   std::vector<int> kernel (9);
   for(int i=0;i<kernel.size();i++) kernel[i] = i;
   int rowsize = 10;
   auto r = in.begin()+15;

   stencil(p, in.begin(), in.end(), out.begin(),
     [&](auto it, std::vector<int> &ng){

      if( (it-in.begin()) > rowsize 
       && (it-in.begin()) < (in.end()-rowsize-in.begin()) 
       && ((it-in.begin()) % rowsize) != 0
       && ((it-in.begin()) % rowsize) != (rowsize-1)) {
         auto val = ng[0]*kernel[0] + ng[1]*kernel[1] + ng[2]*kernel[2] 
                  + ng[3]*kernel[3] +/* *(it) **/ kernel[4] + ng[4]*kernel[5]
                  + ng[5]*kernel[6] + ng[6]*kernel[7] + ng[7]*kernel[8];
//         auto val = *(it);
         return val;
      }
      return *(it);
   },
   [&](auto it){
        std::vector<int> nn;
      if( (it-in.begin()) > rowsize
       && (it-in.begin()) < (in.end()-rowsize-in.begin())
       && ((it-in.begin()) % rowsize) != 0
       && ((it-in.begin()) % rowsize) != (rowsize-1)) {
 
        nn.push_back(*(it-1 - rowsize));
        nn.push_back(*(it - rowsize));
        nn.push_back(*(it-1 - rowsize));
        nn.push_back(*(it+1));
        nn.push_back(*(it-1));
        nn.push_back(*(it-1 + rowsize));
        nn.push_back(*(it + rowsize));
        nn.push_back(*(it-1 + rowsize)); 

     }
        return nn;
   }
);

for(int i=0;i<out.size();i++){
  //if(i%rowsize==0) std::cout<<"\n";
  //std::cout<<out[i]<<"\t";
  output += out[i];

}

return output;


}


TEST(GrPPI, stencil_example1_seq ){
    sequential_execution p{};
    EXPECT_EQ(113782, stencil_example1(p) );
}

TEST(GrPPI, stencil_example1_thr ){
    parallel_execution_native p{NTHREADS};
    EXPECT_EQ(113782, stencil_example1(p) );
}

#ifdef GRPPI_OMP
    TEST(GrPPI, stencil_example1_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(113782, stencil_example1(p) );
    }
#endif
#ifdef GRPPI_TBB
    TEST(GrPPI, stencil_example1_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(113782, stencil_example1(p) );
    }
#endif



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
