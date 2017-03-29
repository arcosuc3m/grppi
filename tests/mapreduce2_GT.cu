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
#include <ppi/mapreduce.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int mapreduce_example1() {
    std::vector<std::vector<int>> mat(10000);
    for(int i=0;i<mat.size();i++) {
        mat[i] = std::vector<int> (10000);
        for(int j=0;j<mat[i].size();j++){
            mat[i][j] = 1;
        }
    }
    std::vector<int> v(10000);
    for( int i= 0 ; i< v.size(); i++){
         v[i] = 2;
    }
    std::vector<int> out(10000);

    auto p = parallel_execution_thrust(1, thrust::cuda::par);
    cudaGetDeviceCount(&(p.num_gpus));

    int result = 0;
    int val = 3;

    MapReduce(p, v.begin(), v.end(), 
           [=] __device__  (int in){ 
                  return val * in;  
           },
           [] __device__ (int in, int out){
		 return out + in; 
           },
           result
   );

   return result;

//    for(int i = 0; i< out.size();i++) std::cout<<"REDUCTION ["<<i<<"] = "<<out[i]<<std::endl;
}

TEST(GrPPI, mapreduce_gpu ){
    EXPECT_EQ(60000, mapreduce_example1() );
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
