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

// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"

void capitalize(grppi::dynamic_execution & e/*, 
                std::istream & in, std::ostream & out*/)
{
  using namespace std;
  using namespace experimental;
  auto cons =    [&](int word) {
      std::cout << word << std::endl;
    };
  
  int n = 10;

 /* grppi::pipeline(e,
    [&]() -> optional<int> {
       n --;
       if(n<0) return {};
       return {n};
    },
    //low pass filter
    [](int a){
      return a+1;
    },
    //FMDemodulator
    [](int a){
      return a+1;
    },
    //Equalizer
    grppi::split_join(grppi::duplicate{},
      grppi::pipeline(
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [](int a){ return a;},
           //Low pass filter
           [](int a){ return a;}
         ), 
         //grppi::window(2);
         //Substracter
         [](std::tuple<int,int> a){
            return a.get<0>-a.get<1>; 
         },
         //Amplify
         [](int a){
            return a;
         }
       )
    ),
    grppi::window(4),
    //Anonfliter
    [](std::tuple<int,int,int,int> a){
        return a.get<0>-a.get<1>; 
    },
    [&](int a){
      std::cout<<a<<std::endl;
    }
  );
    */ 

  grppi::pipeline(e,
    [&]() -> optional<int> {
       n --;
       if(n<0) return {};
       return {n};
    },
    //low pass filter
    [](int a){
      return a+1;
    },
    //FMDemodulator
    [](int a){
      return a+1;
    },
    //Equalizer
    grppi::split_join(grppi::duplicate{},
      grppi::pipeline(
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [](int a){ return a;},
           //Low pass filter
           [](int a){ return a;}
         ), 
         grppi::window(grppi::count_based<int>(2,2)),
         //Substracter
         [](auto a){
            return a[0]-a[1]; 
         },
         //Amplify
         [](int a){
            return a;
         }
       ),
       grppi::pipeline(
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [](int a){ return a;},
           //Low pass filter
           [](int a){ return a;}
         ),
         grppi::window(grppi::count_based<int>(2,2)),
         //grppi::window(2);
         //Substracter
         [](auto a){
            return a[0]-a[1];
         },
         //Amplify
         [](int a){
            return a;
         }
      ),
      grppi::pipeline(
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [](int a){ return a;},
           //Low pass filter
           [](int a){ return a;}
         ),
         grppi::window(grppi::count_based<int>(2,2)),
         //Substracter
         [](auto a){
            return a[0]-a[1];
         },
         //Amplify
         [](int a){
            return a;
         }
       ),
       grppi::pipeline(
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [](int a){ return a;},
           //Low pass filter
           [](int a){ return a;}
         ),
         grppi::window(grppi::count_based<int>(2,2)),
         //Substracter
         [](vector<int> a){
            return a[0]-a[1];
         },
         //Amplify
         [](int a){
            return a;
         }
       )
    ),
    grppi::window(grppi::count_based<int>(2,2)),
    //Anonfliter
    [](auto a){
        return a[0]-a[1]-a[2]-a[3]; 
    },
    [&](int a){
      std::cout<<a<<std::endl;
    }
  );
}

void print_message(const std::string & prog, const std::string & msg) {
  using namespace std;

  cerr << msg << endl;
  cerr << "Usage: " << prog << " mode " << endl;
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

/*  ifstream in{argv[2]};
  if (!in) {
    print_message(argv[0], "Cannot open file "s + argv[2]);
    return -2;
  }

  ofstream out{argv[3]};
  if (!out) {
    print_message(argv[0], "Cannot open file "s + argv[3]);
    return -2;
  }*/

  if (!run_test(argv[1], capitalize/*, in, out*/)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }

  return 0;
}
