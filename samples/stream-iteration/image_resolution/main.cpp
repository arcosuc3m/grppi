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
#include <experimental/optional>

// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void image_resolution(grppi::dynamic_execution & e, int size) {
  using namespace std;
  using namespace experimental;
  using image_t = std::vector<std::vector<short>>;
  int a = 10;
 
  grppi::pipeline(e,
    [&]() -> optional<image_t> {
      a--; 
      if ( a == 0 ) {
         return {};
      }
      else {
        image_t input_image(size, std::vector<short>(size));
        for(int i= 0 ; i < size; i++){
          for(int j= 0 ; j < size; j++){
            input_image[i][j] = rand() % 256;
          }
        }
        return { input_image };
      }
    },
    grppi::repeat_until( 
      []( auto image ) { 
        image_t output_image{ (image.size()/2) , std::vector<short>(image[0].size()/2) }; 
        int currsize = image.size();
        for(int i= 0, ni= 0; i< currsize; i+= 2, ni++) {
          for(int j= 0, nj= 0; j< currsize; j+= 2, nj++) {
            output_image[ni][nj] = 0;
            for(int k = -1; k<2; k++){
              for(int r = -1; r<2; r++){
                if(j+r>0 && j+r<image.size() && i+k>0 && i+k<image.size()){
                  output_image[ni][nj]+= image[i+k][j+r];
                }
              }
            }
          } 
        } 
        return output_image;
      },
      []( auto image ){
        return image.size() < 128;
      },
      []( auto image ){
        return (image.size() == 1024 || 
                image.size() == 512 || 
                image.size() == 128);
      }
    ),
    [&]( auto image ){
      for(int i =0 ; i< image.size(); i++){
        for(int j =0; j< image.size(); j++){
          //std::cout << image[i][j]; 
        }
        //std::cout << std::endl;
      }
      std::cout<<image.size()<<std::endl;
    }
  );
}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " image_size mode" << endl;
  cerr << "  size: Size of the initial image" << endl;
  cerr << "  mode:" << endl;
  print_available_modes(cerr);
}


int main(int argc, char **argv) {
    
  using namespace std;

  if (argc < 3) {
    print_message(argv[0], "Invalid number of arguments.");
    return -1;
  }

  int size = stoi(argv[1]);
  if (size<=0) {
    print_message(argv[0], "Invalid size. Use a positive number.");
    return -1;
  }

  if (!run_test(argv[2], image_resolution, size)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
