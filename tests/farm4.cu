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
#include <ppi/farm.h>



using namespace std;
using namespace grppi;

void farm_example1() {

    int a = 20000;

    auto p = parallel_execution_thrust(1, thrust::cuda::par);
    cudaGetDeviceCount(&(p.num_gpus));

    farm(p,
        // Farm generator as lambda
        [&]() {
            a--; 
            if ( a == 0 ) { 
                return optional< vector<int> >();
            } 
            else {
                return optional< vector<int> >( vector<int>(1000,a) );
            }
        },
        // Farm kernel as lambda
	[] __device__ (int in)->int
           {
               return in;
           },
        [&](vector<int> v){
           std::cout<<v[0]<<"\n";
        }

 
    );
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    farm_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
