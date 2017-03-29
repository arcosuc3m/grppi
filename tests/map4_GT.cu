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
#include <ppi/map.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;

int map_example1() {

    int output = 0;

    std::vector<int> in(1000);
    for(int i=0;i<in.size();i++) in[i] = i;
    std::vector<int> out(1000);

    auto p = parallel_execution_thrust(1, thrust::cuda::par);
    cudaGetDeviceCount(&(p.num_gpus));

    Map(p, in.begin(), in.end(), out.begin(), [] __device__ (int i)->int { return i; });

    for(int i=0;i<in.size();i++){
      output += out[i];  
    } 
    return output;
}

TEST(GrPPI, map_gpu ){
    EXPECT_EQ(499500, map_example1() );
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
