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
// grppi
#include "grppi.h"

// Samples shared utilities
#include "../../util/util.h"





void capitalize(grppi::parallel_execution_native & e/*, 
                std::istream & in, std::ostream & out*/)
{
  using namespace std;
  using namespace experimental;
  
  //INIT FMRADIOCORE
  float rate = 250000000; // 250 MHz sampling rate is sensible
  float cutoff = 108000000; //guess... doesn't FM freq max at 108 Mhz? 
  int taps = 64;
  float max = 27000;
  float bandwidth = 10000;
  int decimation = 4;
  // determine where equalizer cuts.  Note that <eqBands> is the
  // number of CUTS; there are <eqBands>-1 bands, with parameters
  // held in slots 1..<eqBands> of associated arrays.
  int eqBands = 5;
  float eqCutoff[eqBands];
  float eqGain[eqBands];
  float low = 55;
  float high = 1760;
  float pi = 3.1416;

  for (int i=0; i<eqBands; i++) {
     // have exponentially spaced cutoffs
     eqCutoff[i] = (float)exp(i*(log(high)-log(low))/(eqBands-1) + log(low));
  }

  // first gain doesn't really correspond to a band
  eqGain[0] = 0;
  for (int i=1; i<eqBands; i++) {
     // the gain grows linearly towards the center bands
    float val = (((float)(i-1))-(((float)(eqBands-2))/2.0)) / 5.0;
    eqGain[i] = val > 0 ? 2.0-val : 2.0+val;
  }

  //INIT BANDLOWPASS
  std::vector<std::vector<float>> coeffs (eqBands, std::vector<float>(taps,0.0)); 
  for(int b = 0; b< eqBands; b++){
    int i;
    float m = taps - 1;
    float w = 2 * pi * eqCutoff[b] / rate;
    for (i = 0; i < taps; i++) {
      if (i - m/2 == 0)
        coeffs[b][i] = w/pi;
      else
        coeffs[b][i] = sin(w*(i-m/2)) / pi / (i-m/2) *
          (0.54 - 0.46 * cos(2*pi*i/m));
    }
  } 

  //INIT LOW PASS
  float coeff[taps];
  int i;
  float m = taps - 1;
  float w = 2 * pi * cutoff / rate;
  for (i = 0; i < taps; i++) {
    if (i - m/2 == 0)
      coeff[i] = w/pi;
    else
      coeff[i] = sin(w*(i-m/2)) / pi / (i-m/2) *
        (0.54 - 0.46 * cos(2*pi*i/m));
  }

  
  
  //DEMODULATOR INIT
  float mGain = max*(rate/(bandwidth*pi));


  int end = 4*2*64*10;
  float x = 0;

  grppi::pipeline(e,
    [&]() -> optional<float> {
       x++;
       if(x==end) return {};
       return {x};
    },
    grppi::window(grppi::count_based<float>(taps,1+decimation)), //64,4
    //low pass filter
    [&coeff](auto window){
      int i=0;
      float sum = 0;
      for(auto it = window.begin(); it != window.end(); it++){
         sum+= *it*coeff[i++];
      }
      return sum;
    },
    grppi::window(grppi::count_based<float>(2,1)),
    //FMDemodulator
    [&mGain](auto w)->float{
      if (w.size()==2){
        return mGain* atan(w[0] * w[1]);
      }
      return 0.0;
    },
    grppi::window(grppi::count_based<float>(taps,1)), //64,1
    //Equalizer
    grppi::split_join(grppi::duplicate{},
      grppi::pipeline(
        //grppi::window(grppi::count_based<float>(taps,4)), //64,1
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [&coeffs](auto window){
//           [&coeffs](auto window)->float{
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= (*it)*coeffs[0][i++];
              }
              return sum;
           },
           //Low pass filter
           [&coeffs](auto window){
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= *it*coeffs[1][i++];
              }
              return sum;
           }
         ), 
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [](auto w){
            float val = 0.0;
            if(w.size()==2)
              val = w[1]-w[0]; 
            return val;
         },
         //Amplify
         [&eqGain](float a){
            return a * eqGain[1];
         }
       ),
       grppi::pipeline(
        //grppi::window(grppi::count_based<float>(taps,1)), //64,1
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [&coeffs](auto window){
//           [&coeffs](auto window)->float{
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= (*it)*coeffs[1][i++];
              }
              return sum;
           },
           //Low pass filter
           [&coeffs](auto window){
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= *it*coeffs[2][i++];
              }
              return sum;
           }
         ),
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [](auto w){
            float val = 0.0;
            if(w.size()==2)
              val = w[1]-w[0];
            return val;
         },
         //Amplify
         [&eqGain](float a){
            return a * eqGain[2];
         }
       ),
       grppi::pipeline(
        //grppi::window(grppi::count_based<float>(taps,1)), //64,1
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [&coeffs](auto window){
//           [&coeffs](auto window)->float{
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= (*it)*coeffs[2][i++];
              }
              return sum;
           },
           //Low pass filter
           [&coeffs](auto window){
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= *it*coeffs[3][i++];
              }
              return sum;
           }
         ),
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [](auto w){
            float val = 0.0;
            if(w.size()==2)
              val = w[1]-w[0];
            return val;
         },
         //Amplify
         [&eqGain](float a){
            return a * eqGain[3];
         }
       ),
       grppi::pipeline(
//        grppi::window(grppi::count_based<float>(taps,4)), //64,1
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [&coeffs](auto window){
//           [&coeffs](auto window)->float{
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= (*it)*coeffs[3][i++];
              }
              return sum;
           },
           //Low pass filter
           [&coeffs](auto window){
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= *it*coeffs[4][i++];
              }
              return sum;
           }
         ),
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [](auto w){
            float val = 0.0;
            if(w.size()==2)
              val = w[1]-w[0];
            return val;
         },
         //Amplify
         [&eqGain](float a){
            return a * eqGain[4];
         }
       )

    ),
    grppi::window(grppi::count_based<float>(4,4)),
    //Anonfliter
    [](auto a){
       float sum = 0.0;
       int i= 0;
       for ( auto it = a.begin(); it!= a.end(); it++){
          sum+=*it;
       }
       return sum;
    },
    [&](float a){
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
/*
  if (!run_test(argv[1], capitalize, in, out)) {
    print_message(argv[0], "Invalid policy.");
    return -1;
  }
*/
  auto e = grppi::parallel_execution_native{};
  capitalize(e);
  return 0;
}
