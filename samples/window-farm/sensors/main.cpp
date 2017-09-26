/**
* @version    GrPPI v0.1
* @copyright    Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license    GNU/GPL, see LICENSE.txt
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
// Standard library
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <numeric>
#include <stdexcept>
#include <cctype>
#include <cmath>
#include <stdlib.h>
#include <ctime>
#include <math.h>
#include <atomic>
#include <tgmath.h> 
// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"


template <typename E, typename W>
void sensor_analysis(E & e, W window, int ntask/*, 
                std::istream & in, std::ostream & out*/)
{
  using namespace std;
  using namespace experimental;



  int n = 0;
  const struct timespec time[]{{0, 500000L}};

  auto start = std::chrono::high_resolution_clock::now();


  grppi::pipeline( e,
     [&](){
       n++;
       nanosleep(time, NULL);
       if(n != 200000)
          return optional<int>{rand() % 2000}; 
       else
          return optional<int>{};
     },
/*   grppi::window(window),
     grppi::farm(ntask, [](auto window) {
       double value = 0.0;
       for(auto i = 0; i < 10000; i++)
       for( auto it = window.begin(); it != window.end(); it ++){
         double aux = *it; 
         value = sqrt(aux*aux + value*value);
       }
       return value;
     }),*/
     grppi::farm(ntask, [](auto window) {
       double value = 0.0;
       for(auto i = 0; i < 10000; i++)
       for( auto it = window.begin(); it != window.end(); it ++){
         double aux = *it; 
         value = sqrt(aux*aux + value*value);
       }
       return value;
     },
     window),
     [&](double a){
       std::cout<<a<<std::endl;
     }
  );
   
}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " policy num_cities" <<endl;
/*  cerr << "  input: Input file name" << endl;
  cerr << "  output: Output file name" << endl;*/
  cerr << "  mode:" << endl;
  print_available_modes(cerr);
}


int main(int argc, char **argv) {
    
  using namespace std;

  if(argc < 2){
    print_message(argv[0], "Invalid number of arguments.");
    return -1;
  }
  auto e = grppi::parallel_execution_native{};
//auto e = grppi::parallel_execution_omp{};
//auto e = grppi::parallel_execution_tbb{};
//auto e = grppi::sequential_execution{};
  if(string(argv[2]) == "count"){
    auto w = grppi::count_based<int>(atoi(argv[3]),atoi(argv[4]));
    sensor_analysis(e, w, atoi(argv[1]) );
  }
  if(string(argv[2]) == "time"){
    auto w = grppi::time_based<int>(atof(argv[3]),atof(argv[4]));
    sensor_analysis(e, w, atoi(argv[1]) );
  }
  return 0;
}
