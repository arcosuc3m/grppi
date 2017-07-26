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

using namespace std;
using namespace grppi;

void map_example1() {

#ifndef NTHREADS
#define NTHREADS 6
#endif

#ifdef SEQ
    sequential_execution p{};
#elif OMP
    parallel_execution_omp p{};
#elif TBB
    parallel_execution_tbb p{NTHREADS};
#elif THR
    parallel_execution_native p{NTHREADS};
#else
    sequential_execution p{};
#endif

   std::vector<int> in(100);
   for(int i=0;i<in.size();i++) in[i] = i;
   std::vector<int> kernel (9);
   for(int i=0;i<kernel.size();i++) kernel[i] = i;
   std::vector<int> out(100);
   int rowsize = 10;
   auto r = in.begin()+15;
   std::cout << (r-in.begin()) << " " << (in.end()-rowsize-in.begin()) << " " <<((r-in.begin()) % rowsize)<<"\n";

   stencil(p, in.begin(), in.end(), out.begin(),
     [&](auto it, std::vector<int> ng){

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
  if(i%rowsize==0) std::cout<<"\n";
  std::cout<<out[i]<<"\t";
}
std::cout<<std::endl;

}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    map_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
