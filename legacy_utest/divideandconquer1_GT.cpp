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
#include "divideandconquer.h"
#include <gtest/gtest.h>

using namespace std;
using namespace grppi;


int dividec_example1(auto &p) {

#ifndef NTHREADS
#define NTHREADS 6
#endif

    std::vector<int> v(20);
    for( int i= 0 ; i< v.size(); i++){
        v[i] = i;
    }
    int out = 0;
    
    divide_and_conquer(p,v, out,
                     [&](auto & v){
        std::vector<std::vector<int>> subproblem;
        if(v.size() <= 2){ subproblem.push_back(v);return subproblem; }

        std::vector<int> v1 (v.size()/2);
        std::vector<int> v2 ((v.size()+1)/2);
        int i;
        for(i=0;i<v1.size(); i++){
            v1[i] = v[i];
            //std::cout<< "V: "<<v[i]<<"V1 : "<< v1[i] <<"\n";
        }
        for(int j=0;j<v2.size(); i++, j++){
            v2[j]= v[i];
            //std::cout<< "V2 : "<< v2[j] <<"\n";

        }
        subproblem.push_back(v1);
        subproblem.push_back(v2);
        //std::cout << "Divide problem size " << v.size() << " -> " << v1.size() << " " << v2.size() << "\n";
        return subproblem;
    },
    // base case management: vector<int> ->int
    [&](const vector<int> & problem, int & out){
        //                out = 0;
        //std::cout << "Base case problem size " << problem.size() << "\n";
        for(int i= 0; i< problem.size(); i++) out += problem[i];
    },
    // Merge: vector<T> -> T
        [&](auto & partial, auto & out){
        //std::cout<<"MERGE " << partial << "IN " << out<<"\n";
        out += partial;
    }
    );

    return out;

}

TEST(GrPPI, divideandconquer1_seq ){
    sequential_execution p{};
    EXPECT_EQ(190, dividec_example1(p) );
}

TEST(GrPPI, divideandconquer1_thr ){
    parallel_execution_thr p{NTHREADS};
    EXPECT_EQ(190, dividec_example1(p) );
}

#ifdef GRPPI_OMP
    TEST(GrPPI, divideandconquer1_omp ){
        parallel_execution_omp p{NTHREADS};
        EXPECT_EQ(190, dividec_example1(p) );
    }
#endif
#ifdef GRPPI_TBB
    TEST(GrPPI, divideandconquer1_tbb ){
        parallel_execution_tbb p{NTHREADS};
        EXPECT_EQ(190, dividec_example1(p) );
    }
#endif




int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
