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
#include <mapreduce.h>

using namespace std;
using namespace grppi;

void mapreduce_example1() {
  using namespace std;

#ifndef NTHREADS
#define NTHREADS 6
#endif

#ifdef SEQ
    sequential_execution p{};
#elif OMP
    parallel_execution_omp p{NTHREADS};
#elif TBB
    parallel_execution_tbb p{NTHREADS};
#elif THR
    parallel_execution_native p{NTHREADS};
#else
    sequential_execution p{};
#endif

   std::vector<std::string> words{"a","b","a","c","d","e","c","c","a","b"};
   std::map<std::string,int> init;
   auto result = map_reduce(
      p,
      words.begin(),
      words.end(),
      init,
      [](std::string word){
         std::map<string,int> key_value;
         key_value[word]=1;
         return key_value;
      },
      [](auto map1,auto map2){
          for(auto & i : map2) {
             map1[i.first] += i.second;
          }
          return map1;

      }
   );
   std::cout<<"Word : count "<<std::endl;
   for(auto & i : result) {
     std::cout<< i.first << " : " << i.second<<std::endl;
   }
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    mapreduce_example1();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
