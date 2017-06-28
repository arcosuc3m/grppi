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
#include <pipeline.h>
#include <farm.h>
#include <algorithm>

using namespace std;
using namespace grppi;

void pipeline_farm_example() {

#ifndef NTHREADS
#define NTHREADS 6
#endif

#ifdef SEQ
    sequential_execution p{}, f{};
#elif OMP
    parallel_execution_omp p{3}, f{NTHREADS-3};
#elif TBB
    parallel_execution_tbb p{3}, f{NTHREADS-3};
#elif THR
    parallel_execution_native p{3}, f{NTHREADS-3};
#else
    sequential_execution p{}, f{};
#endif

    int n=10;
    std::vector<string> output;
    p.ordering=true;

    auto f_obj = farm(f,
         [&](std::vector<int> v) {
  //      std::cout<<"FARM THREAD ID : "<<f.get_threadID()<<std::endl;
        //std::cout << "Worker del farm\n";
        std::vector<long> acumm( v.size() );
        for(unsigned i = 0; i < acumm.size(); i++ ){
            acumm[i] = 0;
            for(auto j : v){
                acumm[i] += j;
            }
        }
        return (acumm);
    }
    );

     auto gen = [&]() {
//        std::cout<<"PIPE THREAD ID : "<<p.get_threadID()<<std::endl;
          std::vector<int> v(5);
          for ( int i = 0; i < 5; i++ )
             v[ i ] = i + n;

          if ( n < 0 )
               return optional< std::vector<int> >();
           n--;
           return optional<std::vector<int>>(v);
    };
    pipeline(p, [&]() {
//        std::cout<<"PIPE THREAD ID : "<<p.get_threadID()<<std::endl;
          std::vector<int> v(5);
          for ( int i = 0; i < 5; i++ )
             v[ i ] = i + n;

          if ( n < 0 )
               return optional< std::vector<int> >();
           n--;
           return optional<std::vector<int>>(v);
    },
    farm(f,
         [&](std::vector<int> v) {
  //      std::cout<<"FARM THREAD ID : "<<f.get_threadID()<<std::endl;
        //std::cout << "Worker del farm\n";
        std::vector<long> acumm( v.size() );
        for(unsigned i = 0; i < acumm.size(); i++ ){
            acumm[i] = 0;
            for(auto j : v){
                acumm[i] += j;
            }
        }
        return (acumm);
    }
    ),
//    f_obj,
            // Pipeline stage 2
            [&]( std::vector<long> acc ) {
    //    std::cout<<"PIPE THREAD ID : "<<p.get_threadID()<<std::endl;
        //std::cout << "Stadio2\n";
        double acumm = 0;
        for ( int i = 0; i < acc.size(); i++ )
            acumm += acc[ i ];

        return acumm;
    },

    // Pipeline stage 3
    [&]( double v ) {
     //   std::cout<<"PIPE THREAD ID : "<<p.get_threadID()<<std::endl;
        //std::cout << "Stadio3\n";
        //std::cout << v << std::endl;
        output.push_back("Stadio3 " + std::to_string(v) );
    }
    );

pipeline(p, [&]() {
//        std::cout<<"PIPE THREAD ID : "<<p.get_threadID()<<std::endl;
          std::vector<int> v(5);
          for ( int i = 0; i < 5; i++ )
             v[ i ] = i + n;

          if ( n < 0 )
               return optional< std::vector<int> >();
           n--;
           return optional<std::vector<int>>(v);
    },
    f_obj,
            // Pipeline stage 2
            [&]( std::vector<long> acc ) {
    //    std::cout<<"PIPE THREAD ID : "<<p.get_threadID()<<std::endl;
        //std::cout << "Stadio2\n";
        double acumm = 0;
        for ( int i = 0; i < acc.size(); i++ )
            acumm += acc[ i ];

        return acumm;
    },

    // Pipeline stage 3
    [&]( double v ) {
     //   std::cout<<"PIPE THREAD ID : "<<p.get_threadID()<<std::endl;
        //std::cout << "Stadio3\n";
        //std::cout << v << std::endl;
        output.push_back("Stadio3 " + std::to_string(v) );
    }
    ); 

    // Sort to constant result
    std::sort(output.begin(), output.end());
    // Print results
    for (int i = 0; i < output.size(); i++){
        std::cout << output[i] << std::endl;
    }
}

int main() {

    //$ auto start = std::chrono::high_resolution_clock::now();
    pipeline_farm_example();
    //$ auto elapsed = std::chrono::high_resolution_clock::now() - start;

    //$ long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count();
    //$ std::cout << "Execution time : " << microseconds << " us" << std::endl;

    return 0;
}
