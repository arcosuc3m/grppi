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


//#include "extrae_user_events.h"
//#include <extrae.h>

template <typename E>
void equalizer(E & e, int num_samples, int num_taps)
{
  using namespace std;
  using namespace experimental;
  
  //INIT FMRADIOCORE
  float rate = 250000000; // 250 MHz sampling rate is sensible
  float cutoff = 108000000; //guess... doesn't FM freq max at 108 Mhz? 
  int taps = num_taps;
  float max = 27000;
  float bandwidth = 10000;
  int decimation = taps/16;
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


  //int end = 4*2*64*10;
  int end = num_samples;
  float x = 0;
  int outs=0;
  //Extrae_init();
  auto start = std::chrono::high_resolution_clock::now();
  grppi::pipeline(e,
    [&]() -> optional<float> {
       //Extrae_event(60000000,3);
       x++;
       if(x==end) return {};
       //Extrae_event(60000000,0);
       return {x};
    },
    grppi::window(grppi::count_based<float>(taps,1+decimation)), //64,4
    //low pass filter
    [&coeff](auto window){
      //Extrae_event(60000000,4);
      int i=0;
      float sum = 0;
      for(auto it = window.begin(); it != window.end(); it++){
         sum+= *it*coeff[i++];
      }
      //Extrae_event(60000000,0);
      return sum;
    },
    grppi::window(grppi::count_based<float>(2,1)),
    //FMDemodulator
    [&mGain](auto w)->float{
      //Extrae_event(60000000,5);
      if (w.size()==2){
        //Extrae_event(60000000,0);
        return mGain* atan(w[0] * w[1]);
      }
      //Extrae_event(60000000,0);
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
              //Extrae_event(60000000,6);
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= (*it)*coeffs[0][i++];
              }
              //Extrae_event(60000000,0);
              return sum;
            },
           //Low pass filter
           [&coeffs](auto window){
             //Extrae_event(60000000,7);
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= *it*coeffs[1][i++];
              }
              //Extrae_event(60000000,0);
              return sum;
           }
         ), 
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [&eqGain](auto w){
            //Extrae_event(60000000,8);
            float val = 0.0;
            if(w.size()==2)
              val = w[1]-w[0]; 
        /*    return val;
         },
         //Amplify
         [&eqGain](float a){*/
            //Extrae_event(60000000,0);
            return val * eqGain[1];
         }
       ),
       grppi::pipeline(
        //grppi::window(grppi::count_based<float>(taps,1)), //64,1
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [&coeffs](auto window){
//           [&coeffs](auto window)->float{
              //Extrae_event(60000000,9);
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= (*it)*coeffs[1][i++];
              }
              //Extrae_event(60000000,0);
              return sum;
           },
           //Low pass filter
           [&coeffs](auto window){
              //Extrae_event(60000000,10);
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= *it*coeffs[2][i++];
              }
//Extrae_event(60000000,0);
              return sum;
           }
         ),
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [&eqGain](auto w){
 //Extrae_event(60000000,11);
            float val = 0.0;
            if(w.size()==2)
              val = w[1]-w[0];
         /*   return val;
         },
         //Amplify
         [&eqGain](float a){*/
 //Extrae_event(60000000,0);
            return val * eqGain[2];
         }
       ),
       grppi::pipeline(
        //grppi::window(grppi::count_based<float>(taps,1)), //64,1
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [&coeffs](auto window){
//           [&coeffs](auto window)->float{
//Extrae_event(60000000,12);
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= (*it)*coeffs[2][i++];
              }
//Extrae_event(60000000,0);
              return sum;
           },
           //Low pass filter
           [&coeffs](auto window){
//Extrae_event(60000000,13);
              int i=0;
              float sum = 0.0;
              for(auto it = window.begin(); it != window.end(); it++){
                 sum+= *it*coeffs[3][i++];
              }
//Extrae_event(60000000,0);
              return sum;
           }
         ),
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [&eqGain](auto w){
//Extrae_event(60000000,14);
            float val = 0.0;
            if(w.size()==2)
              val = w[1]-w[0];
        /*    return val;
         },
         //Amplify
         [&eqGain](float a){*/
//Extrae_event(60000000,0);
            return val * eqGain[3];
         }
      /* ),
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
         [&eqGain](auto w){
            float val = 0.0;
            if(w.size()==2)
              val = w[1]-w[0];
            return val;
         },
         //Amplify
         [&eqGain](float a){
            return val * eqGain[4];
         }*/
       )

    ),
    grppi::window(grppi::count_based<float>(3,3)),
    //Anonfliter
    [](auto a){
//Extrae_event(60000000,15);
       float sum = 0.0;
       int i= 0;
       for ( auto it = a.begin(); it!= a.end(); it++){
          sum+=*it;
       }
//Extrae_event(60000000,0);
       return sum;
  /*  },
    [&](float a){
      //std::cout<<a<0std::endl;
      outs++;*/
    }
 );

  auto endt = std::chrono::high_resolution_clock::now();
  int elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (endt-start).count();
//Extrae_fini();
  std::cout << elapsed_time <<std::endl;
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
  if(string(argv[1]) == "thr"){
    auto e = grppi::parallel_execution_native{};
    e.set_queue_attributes(100,grppi::queue_mode::lockfree);
    equalizer(e,atoi(argv[2]),atoi(argv[3]));
  }
  if(string(argv[1]) == "seq"){
    auto e = grppi::sequential_execution{};
    equalizer(e,atoi(argv[2]),atoi(argv[3]));
  }
  if(string(argv[1]) == "omp"){
    auto e = grppi::parallel_execution_omp{};
    e.set_queue_attributes(100,grppi::queue_mode::lockfree);
    equalizer(e,atoi(argv[2]),atoi(argv[3]));
  }
  return 0;
}
