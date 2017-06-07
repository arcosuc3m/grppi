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
#include <string>
#include "map.h"


void map_example1(int n, auto &p) {

		using namespace std;
		
    std::vector<int> in(n);
    std::iota(in.begin(), in.end(), 0);
    std::vector<int> out(n);
    
    grppi::map(p, in.begin(), in.end(), out.begin(),
         [](int i) { return i*2; } );

		std::copy(begin(out), end(out), ostream_iterator<int>(cout, " "));
		std::cout << std::endl;

}

int main(int argc, char **argv) {

    if(argc < 3){
        std::cout << "Invalid number of arguments. Usage: ./binary <problem_size> <SEQ|THR|TBB|OMP>" << std::endl;
        return -1;
	  }

    int n = stoi(argv[1]);
    if(n <= 0){
       std::cout << "Invalid problem size. Use a positive integer number." << std::endl;
        return -1;
    }

    if(!strcmp("SEQ", argv[2])){
        grppi::sequential_execution p{};
        map_example1(n,p);
    }else if (!strcmp("THR", argv[2])){
        grppi::parallel_execution_thr p{};
        map_example1(n,p);
    }else if (!strcmp("TBB", argv[2])){
        grppi::parallel_execution_tbb p{};
        map_example1(n,p);
    }else if (!strcmp("OMP", argv[2])){
        grppi::parallel_execution_omp p{};
        map_example1(n,p);
    }else{
        std::cout << "Invalid policy. Usage: ./binary <problem_size> <SEQ|THR|TBB|OMP>" << std::endl;
        return -1;
    }

    return 0;
}
