/*
 * Copyright 2018 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#include "grppi/grppi.h"

// Samples shared utilities
#include "../../util/util.h"

std::vector<std::vector<int>> generate_problem(int num_cities) {
  std::vector<std::vector<int>> problem(num_cities, std::vector<int>(num_cities,0));

  for (int i= 0; i<num_cities; i++)
     for (int j= 0; j<num_cities; j++)
       if (i!=j) {
         int cost = rand() % 10;
         problem[i][j] = cost;
         problem[j][i] = cost;
       }  
   return problem;
}

void print_individual(std::vector<int> individual, int cost) {
  std::cout << "Solution : ";
  for (auto it= individual.begin(); it!= individual.end(); it++)
    std::cout << (*it)+1 << " ";
  std::cout << "Cost : " << cost << std::endl;
}

int compute_cost(std::vector<int> individual, std::vector<std::vector<int>> traveling_cost) {
  int cost = 0;
  for (auto it = individual.begin(); it != individual.end()-1; it++ )
    cost += traveling_cost[(*it)][(*(it+1))];
  return cost;
}

void travel(grppi::dynamic_execution & e, int num_cities) {
  using namespace std;
  using namespace experimental;
  auto problem = generate_problem(num_cities);

  int popsize = 500;
  std::vector<std::vector<int>> population;
  for (auto i= 0; i<popsize; i++){
     std::vector<int> cities;
     for (int i= 0; i<num_cities; i++) 
       cities.push_back(i);
     std::vector<int> individual;
     for (int j= 0; j< num_cities; j++) {
       int c= rand() % (num_cities-j);
       individual.push_back(cities[c]);
       cities.erase(cities.begin()+c);
     }
     population.push_back(individual);
  }
  std::atomic<int> count{0};
  
  grppi::stream_pool(e, population, 
    [] (auto & population) {
      auto individual = population[0];
      population.erase(population.begin());
      return individual;
    },
    [&] (auto individual){
      for(int i= 0; i<num_cities;i++){
        auto cost = compute_cost(individual,problem);
        int swap = rand() % (individual.size()-1);
        auto aux = individual[swap];
        individual[swap] = individual[swap+1];
        individual[swap+1] = aux;
        if(cost < compute_cost(individual,problem)){
          individual[swap+1] = individual[swap];
          individual[swap+1] = aux;
        }
      }
      return individual;
    },
    [&problem] (auto evolved, auto selected){
      return compute_cost(evolved, problem) < compute_cost(selected, problem) ?
             evolved : selected;
    },
    [&problem, &count, num_cities] (auto )
    {
      return 2000 < count++;
    }
  );
  
  auto &best = population[0];
  auto best_cost = compute_cost(best, problem);
  for (auto &ind : population) {
    auto aux = compute_cost(ind,problem);
    if( aux < best_cost ) {
      best = ind;
      best_cost = aux;
    }
  }
  print_individual(best, best_cost);
}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " num_cities mode" <<endl;
  cerr << "  num_cities: Number of cities in the TSP problem" << endl;
  cerr << "  mode:" << endl;
  print_available_modes(cerr);
}

int main(int argc, char **argv) {
    
  using namespace std;

  if(argc < 3){
    print_message(argv[0], "Invalid number of arguments.");
    return -1;
  }

  int num_cities = stoi(argv[1]);
  if (num_cities<=0) {
    print_message(argv[0], "Invalid number of cities. Use a positive number.");
    return -1;
  }

  if (!run_test(argv[2], travel, num_cities)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}

