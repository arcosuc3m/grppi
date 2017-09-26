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
// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"


std::vector<std::vector<int>> generate_problem(int num_cities)
{
   std::vector<std::vector<int>> problem(num_cities, std::vector<int>(num_cities,0));

   for(int i= 0; i < num_cities; i++)
   {
      for(int j = 0; j < num_cities; j++)
      {
        if( i!=j ){
          int cost = rand() % 10;
          problem[i][j] = cost;
          problem[j][i] = cost;
        }  
      }
   }
   return problem;
}


void print_individual(std::vector<int> individual, int cost){
  std::cout<<"SOLUTION : ";
  for(auto it = individual.begin(); it!= individual.end(); it++) std::cout<<(*it)+1<<" ";
  std::cout<<"COST : "<<cost<<std::endl;
}


int computeCost(std::vector<int> individual, std::vector<std::vector<int>> travelingCost){
        int cost = 0;
        for(auto it = individual.begin(); it != individual.end()-1; it++){
            cost += travelingCost[(*it)][(*(it+1))];
        }
        return cost;
}

template <typename E>
void capitalize(E & e, int num_cities/*, 
                std::istream & in, std::ostream & out*/)
{
  using namespace std;
  using namespace experimental;
  srand((unsigned)time(NULL));

  auto problem = generate_problem(num_cities);

  int popsize = 100;
  std::vector<std::vector<int>> population;
  for(auto i = 0; i<popsize;i++){
     std::vector<int> cities;
     for( int i = 0; i< num_cities; i++) cities.push_back(i);
     std::vector<int> individual;
     for( int j = 0; j< num_cities;j++)
     {
        int c = rand() % (num_cities-j);
        individual.push_back(cities[c]);
        cities.erase(cities.begin()+c);
     }
     population.push_back(individual);
  }
  int lastCost = 1000;
  std::atomic<int> count{0};
  bool first = true;
  int NITER = 1000;
  
  auto start = std::chrono::high_resolution_clock::now();

  e.stream_pool(population, 
     [](auto & population) {
       auto individual = population[0];
       population.erase(population.begin());
       return individual;
     },
     [num_cities](auto individual){
       for(int i= 0; i<num_cities;i++){
         int swap = rand() % (individual.size()-1);
         auto aux = individual[swap];
         individual[swap] = individual[swap+1];
         individual[swap+1] = aux;
         auto k = aux;
         for(int j = 0;j<3;j++){
           k = pow(k,k);
           k =sqrt(k);
         }

       }
       return individual;
     },
     [&problem](auto evolved, auto selected){
       auto a = computeCost(evolved, problem);
       auto b = computeCost(selected, problem);
       return a < b ? evolved : selected;
     },
     [&problem, &count, num_cities](auto evolved)
     {
        
        /*if (computeCost(evolved, problem) < num_cities*3){
           print_individual(evolved, computeCost(evolved, problem) );
           return true;
        }*/
        if(200 < count++) return true;
        return false;
     }
  );
}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " mode num_cities" <<endl;
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

  auto e = grppi::parallel_execution_native{24};
  //auto e = grppi::parallel_execution_omp{};
  //auto e = grppi::parallel_execution_tbb{};
  //auto e = grppi::sequential_execution{};
  capitalize(e, atoi(argv[1]) );
  return 0;
}
