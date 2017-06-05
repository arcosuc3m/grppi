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
#include <ppi/farm.h>
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;


int farm_cpu() {

    int out = 0;

    auto tasks = 8;
    auto size = 32 * 1024 * 1024;

    parallel_execution_thr p {8};
    farm(p,
        // Farm generator as lambda
        [&]() {
            tasks--; 
            if ( tasks == 0 ) { 
                return optional< vector<float> >();
            } 
            else {
                return optional< vector<float> >( vector<float>( size, 1.0f) );
            }
        },
        // Farm kernel as lambda
	[] (vector<float> in)->vector<float>
           {
	       vector <float> out(in.size());
	       for (auto i = 0; i < in.size(); ++i) 
                    out[i] = in[i] + sqrt( (in[i] * 2.0f) + (in[i] * 2.0f));
               return out;
           },
        [&](vector<float> v){
            std::cout<<v[0]<<"\n";
            out = v[0];
        } 
    );


    return out;
}


float farm_gpu() {

    float out = 0;

    auto tasks = 8;
    auto size = 32 * 1024 * 1024;

    auto p = parallel_execution_thrust(1, thrust::cuda::par);
    cudaGetDeviceCount(&(p.num_gpus));
    farm(p,
        // Farm generator as lambda
        [&]() {
            tasks--;
            if ( tasks == 0 ) {
                return optional< vector<float> >();
            }
            else {
                return optional< vector<float> >( vector<float>( size, 1.0f) );
            }
        },
        // Farm kernel as lambda
        [] __device__ (float in) -> float 
           {
              return  in + sqrt( (in * 2.0f) + (in * 2.0f));
           },
        [&](vector<float> v){
          //std::cout<<v[0]<<"\n";
          out = v[0];
        }
    );
    return out;
}



TEST(GrPPI, bench_farm_cpu ){
    EXPECT_FLOAT_EQ(3, farm_cpu() );
}

TEST(GrPPI, bench_farm_gpu ){
    EXPECT_FLOAT_EQ(3, farm_gpu() );
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
