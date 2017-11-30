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

constexpr int eqBands = 5;
constexpr float gain(int i){
    float val = (((float)(i-1))-(((float)(eqBands-2))/2.0)) / 5.0;
    return val > 0 ? 2.0-val : 2.0+val;
}

constexpr float eqGain[eqBands]={0,gain(1),gain(2),gain(3),gain(4)};

constexpr int taps = 64; 
constexpr float low = 55;
constexpr float high = 1760;
constexpr float pi = 3.1416;

constexpr  float cutoff = 108000000; //guess... doesn't FM freq max at 108 Mhz? 
constexpr float max = 27000;
constexpr float bandwidth = 10000;
constexpr float rate = 250000000; // 250 MHz sampling rate is sensible

constexpr float cut(int i){
  return (float)exp(i*(log(high)-log(low))/(eqBands-1) + log(low));
}

constexpr float eqCutoff[eqBands]={cut(0),cut(1),cut(2),cut(3),cut(4)}; 




constexpr float coef(int i, int b){
    float m = taps - 1;
    float w = 2 * pi * eqCutoff[b] / rate;
    if (i - m/2 == 0)
       return   w/pi;
    
    return sin(w*(i-m/2)) / pi / (i-m/2) *
          (0.54 - 0.46 * cos(2*pi*i/m));
}

constexpr float coef(int i) {
  float m = taps - 1;
  float w = 2 * pi * cutoff / rate;
  if (i - m/2 == 0)
    return  w/pi;
  return sin(w*(i-m/2)) / pi / (i-m/2) *
        (0.54 - 0.46 * cos(2*pi*i/m));
}

constexpr float coeff[taps] = {coef(0),coef(1),coef(2),coef(3),coef(4),coef(5),coef(6),coef(7),coef(8),coef(9),coef(10),coef(11),coef(12),coef(13),coef(14),coef(15),coef(16),coef(17),coef(18),coef(19),coef(20),coef(21),coef(22),coef(23),coef(24),coef(25),coef(26),coef(27),coef(28),coef(29),coef(30),coef(31),coef(32),coef(33),coef(34),coef(35),coef(36),coef(37),coef(38),coef(39),coef(40),coef(41),coef(42),coef(43),coef(44),coef(45),coef(46),coef(47),coef(48),coef(49),coef(50),coef(51),coef(52),coef(53),coef(54),coef(55),coef(56),coef(57),coef(58),coef(59),coef(60),coef(61),coef(62),coef(63)};

constexpr float coeff_0[taps] = {coef(0,0),coef(1,0),coef(2,0),coef(3,0),coef(4,0),coef(5,0),coef(6,0),coef(7,0),coef(8,0),coef(9,0),coef(10,0),coef(11,0),coef(12,0),coef(13,0),coef(14,0),coef(15,0),coef(16,0),coef(17,0),coef(18,0),coef(19,0),coef(20,0),coef(21,0),coef(22,0),coef(23,0),coef(24,0),coef(25,0),coef(26,0),coef(27,0),coef(28,0),coef(29,0),coef(30,0),coef(31,0),coef(32,0),coef(33,0),coef(34,0),coef(35,0),coef(36,0),coef(37,0),coef(38,0),coef(39,0),coef(40,0),coef(41,0),coef(42,0),coef(43,0),coef(44,0),coef(45,0),coef(46,0),coef(47,0),coef(48,0),coef(49,0),coef(50,0),coef(51,0),coef(52,0),coef(53,0),coef(54,0),coef(55,0),coef(56,0),coef(57,0),coef(58,0),coef(59,0),coef(60,0),coef(61,0),coef(62,0),coef(63,0)};

constexpr float coeff_1[taps] ={coef(0,1),coef(1,1),coef(2,1),coef(3,1),coef(4,1),coef(5,1),coef(6,1),coef(7,1),coef(8,1),coef(9,1),coef(10,1),coef(11,1),coef(12,1),coef(13,1),coef(14,1),coef(15,1),coef(16,1),coef(17,1),coef(18,1),coef(19,1),coef(20,1),coef(21,1),coef(22,1),coef(23,1),coef(24,1),coef(25,1),coef(26,1),coef(27,1),coef(28,1),coef(29,1),coef(30,1),coef(31,1),coef(32,1),coef(33,1),coef(34,1),coef(35,1),coef(36,1),coef(37,1),coef(38,1),coef(39,1),coef(40,1),coef(41,1),coef(42,1),coef(43,1),coef(44,1),coef(45,1),coef(46,1),coef(47,1),coef(48,1),coef(49,1),coef(50,1),coef(51,1),coef(52,1),coef(53,1),coef(54,1),coef(55,1),coef(56,1),coef(57,1),coef(58,1),coef(59,1),coef(60,1),coef(61,1),coef(62,1),coef(63,1)};

constexpr float coeff_2[taps] = {coef(0,2),coef(1,2),coef(2,2),coef(3,2),coef(4,2),coef(5,2),coef(6,2),coef(7,2),coef(8,2),coef(9,2),coef(10,2),coef(11,2),coef(12,2),coef(13,2),coef(14,2),coef(15,2),coef(16,2),coef(17,2),coef(18,2),coef(19,2),coef(20,2),coef(21,2),coef(22,2),coef(23,2),coef(24,2),coef(25,2),coef(26,2),coef(27,2),coef(28,2),coef(29,2),coef(30,2),coef(31,2),coef(32,2),coef(33,2),coef(34,2),coef(35,2),coef(36,2),coef(37,2),coef(38,2),coef(39,2),coef(40,2),coef(41,2),coef(42,2),coef(43,2),coef(44,2),coef(45,2),coef(46,2),coef(47,2),coef(48,2),coef(49,2),coef(50,2),coef(51,2),coef(52,2),coef(53,2),coef(54,2),coef(55,2),coef(56,2),coef(57,2),coef(58,2),coef(59,2),coef(60,2),coef(61,2),coef(62,2),coef(63,2)};

constexpr float coeff_3[taps] = {coef(0,3),coef(1,3),coef(2,3),coef(3,3),coef(4,3),coef(5,3),coef(6,3),coef(7,3),coef(8,3),coef(9,3),coef(10,3),coef(11,3),coef(12,3),coef(13,3),coef(14,3),coef(15,3),coef(16,3),coef(17,3),coef(18,3),coef(19,3),coef(20,3),coef(21,3),coef(22,3),coef(23,3),coef(24,3),coef(25,3),coef(26,3),coef(27,3),coef(28,3),coef(29,3),coef(30,3),coef(31,3),coef(32,3),coef(33,3),coef(34,3),coef(35,3),coef(36,3),coef(37,3),coef(38,3),coef(39,3),coef(40,3),coef(41,3),coef(42,3),coef(43,3),coef(44,3),coef(45,3),coef(46,3),coef(47,3),coef(48,3),coef(49,3),coef(50,3),coef(51,3),coef(52,3),coef(53,3),coef(54,3),coef(55,3),coef(56,3),coef(57,3),coef(58,3),coef(59,3),coef(60,3),coef(61,3),coef(62,3),coef(63,3)};

constexpr float coeff_4[taps] = {coef(0,4),coef(1,4),coef(2,4),coef(3,4),coef(4,4),coef(5,4),coef(6,4),coef(7,4),coef(8,4),coef(9,4),coef(10,4),coef(11,4),coef(12,4),coef(13,4),coef(14,4),coef(15,4),coef(16,4),coef(17,4),coef(18,4),coef(19,4),coef(20,4),coef(21,4),coef(22,4),coef(23,4),coef(24,4),coef(25,4),coef(26,4),coef(27,4),coef(28,4),coef(29,4),coef(30,4),coef(31,4),coef(32,4),coef(33,4),coef(34,4),coef(35,4),coef(36,4),coef(37,4),coef(38,4),coef(39,4),coef(40,4),coef(41,4),coef(42,4),coef(43,4),coef(44,4),coef(45,4),coef(46,4),coef(47,4),coef(48,4),coef(49,4),coef(50,4),coef(51,4),coef(52,4),coef(53,4),coef(54,4),coef(55,4),coef(56,4),coef(57,4),coef(58,4),coef(59,4),coef(60,4),coef(61,4),coef(62,4),coef(63,4)};

//constexpr float coeff_5[taps] = {coef(0,5),coef(1,5),coef(2,5),coef(3,5),coef(4,5),coef(5,5),coef(6,5),coef(7,5),coef(8,5),coef(9,5),coef(10,5),coef(11,5),coef(12,5),coef(13,5),coef(14,5),coef(15,5),coef(16,5),coef(17,5),coef(18,5),coef(19,5),coef(20,5),coef(21,5),coef(22,5),coef(23,5),coef(24,5),coef(25,5),coef(26,5),coef(27,5),coef(28,5),coef(29,5),coef(30,5),coef(31,5),coef(32,5),coef(33,5),coef(34,5),coef(35,5),coef(36,5),coef(37,5),coef(38,5),coef(39,5),coef(40,5),coef(41,5),coef(42,5),coef(43,5),coef(44,5),coef(45,5),coef(46,5),coef(47,5),coef(48,5),coef(49,5),coef(50,5),coef(51,5),coef(52,5),coef(53,5),coef(54,5),coef(55,5),coef(56,5),coef(57,5),coef(58,5),coef(59,5),coef(60,5),coef(61,5),coef(62,5),coef(63,5)};

constexpr float mGain = max*(rate/(bandwidth*pi));

template <typename E>
void equalizer(E & e, int num_samples, int num_taps)
{
  using namespace std;
  using namespace experimental;
  
  //INIT FMRADIOCORE
  constexpr int decimation = taps/16;
//  constexpr int decimation = 128;
  // determine where equalizer cuts.  Note that <eqBands> is the
  // number of CUTS; there are <eqBands>-1 bands, with parameters
  /*for (int i=0; i<eqBands; i++) {
     // have exponentially spaced cutoffs
     eqCutoff[i] = (float)exp(i*(log(high)-log(low))/(eqBands-1) + log(low));
  }
*/
  // first gain doesn't really correspond to a band
/*  eqGain[0] = 0;
  for (int i=1; i<eqBands; i++) {
     // the gain grows linearly towards the center bands
    float val = (((float)(i-1))-(((float)(eqBands-2))/2.0)) / 5.0;
    eqGain[i] = val > 0 ? 2.0-val : 2.0+val;
  }
*/
  //INIT BANDLOWPASS
/*  std::vector<std::vector<float>> coeffs (eqBands, std::vector<float>(taps,0.0)); 
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
*/
  //INIT LOW PASS
  /*float coeff[taps];
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
*/
  
  
  //DEMODULATOR INIT


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
    [](auto && window){
      //Extrae_event(60000000,4);
      float sum = 0;
      for(auto it = 0; it < 64; it++){
         sum+= window[it]*coeff[it];
      }
      //Extrae_event(60000000,0);
      return sum;
    },
    grppi::window(grppi::count_based<float>(2,1)),
    //FMDemodulator
    [](auto && w)->float{
      //Extrae_event(60000000,5);
      //Extrae_event(60000000,0);
      return mGain* atan(w[0] * w[1]);
      //Extrae_event(60000000,0);
    },
    //grppi::window(grppi::count_based<float>(taps,1)), //64,1
    //Equalizer
    grppi::split_join(grppi::duplicate{},
      grppi::pipeline(
        [](auto && p) {return p;},
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           grppi::pipeline(
           grppi::window(grppi::count_based<float>(taps,1)), //64,1
           //Low pass filter
           [](auto && window){
//           [&coeffs](auto window)->float{
              //Extrae_event(60000000,6);
              float sum = 0.0;
              for(auto it = 0; it < 64; it++){
                 sum+= window[it]*coeff_0[it];
              }
              return sum;
            }),
           grppi::pipeline(
           grppi::window(grppi::count_based<float>(taps,1)), //64,1
           //Low pass filter
           [](auto && window){
             //Extrae_event(60000000,7);
              float sum = 0.0;
              for(auto it = 0; it < 64; it++){
                 sum+= window[it]*coeff_1[it];
              }
              //Extrae_event(60000000,0);
              return sum;
           })
         ), 
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [](auto &&w){
            //Extrae_event(60000000,8);
            float val = w[1]-w[0]; 
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
           grppi::pipeline(
           grppi::window(grppi::count_based<float>(taps,1)), //64,1
           //Low pass filter
           [](auto &&window){
//           [&coeffs](auto window)->float{
              //Extrae_event(60000000,9);
              float sum = 0.0;
              for(auto it = 0; it < 64; it++){
                 sum+= window[it]*coeff_1[it];
              }
              //Extrae_event(60000000,0);
              return sum;
           }),
           grppi::pipeline(
           grppi::window(grppi::count_based<float>(taps,1)), //64,1
           //Low pass filter
           [](auto &&window){
              //Extrae_event(60000000,10);
              float sum = 0.0;
              for(auto it = 0; it < 64; it++){
                 sum+= window[it]*coeff_2[it];
              }
//Extrae_event(60000000,0);
              return sum;
           })
         ),
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [](auto &&w){
 //Extrae_event(60000000,11);
            float val = w[1]-w[0];
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
           grppi::pipeline(
           grppi::window(grppi::count_based<float>(taps,1)), //64,1
           //Low pass filter
           [](auto &&window){
//           [&coeffs](auto window)->float{
//Extrae_event(60000000,12);
              float sum = 0.0;
              for(auto it = 0; it < 64; it++){
                 sum+= window[it]*coeff_2[it];
              }
//Extrae_event(60000000,0);
              return sum;
           }),
           grppi::pipeline(
           grppi::window(grppi::count_based<float>(taps,1)), //64,1
           //Low pass filter
           [](auto &&window){
//Extrae_event(60000000,13);
              float sum = 0.0;
              for(auto it = 0; it < 64; it++){
                 sum+= window[it]*coeff_3[it];
              }
//Extrae_event(60000000,0);
              return sum;
           })
         ),
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [](auto &&w){
//Extrae_event(60000000,14);
             float val = w[1]-w[0];
        //    return val;
     //    },
         //Amplify
     //    [&eqGain](float a){
//Extrae_event(60000000,0);
            //std::cout<<val<<"*"<<eqGain[3]<<"="<<val * eqGain[3]<<std::endl;
            return val * eqGain[3];
         }
       ),
       grppi::pipeline(
        grppi::window(grppi::count_based<float>(taps,1)), //64,1
        //Band_pass_filter 
        grppi::split_join(grppi::duplicate{},
           //Low pass filter
           [](auto window){
//           [&coeffs](auto window)->float{
              float sum = 0.0;
              for(auto it = 0; it < 64; it++){
                 sum+= window[it]*coeff_3[it];
              }
              return sum;
           },
           //Low pass filter
           [](auto window){
              float sum = 0.0;
              for(auto it = 0; it < 64; it++){
                 sum+= window[it]*coeff_4[it];
              }
              return sum;
           }
         ),
         grppi::window(grppi::count_based<float>(2,2)),
         //Substracter
         [](auto w){
            float val = w[1]-w[0];
            return val * eqGain[4];
         }
       )
    ),
    grppi::window(grppi::count_based<float>(4,4)),
    //Anonfliter
    [&](auto &&a){
//Extrae_event(60000000,15);
       float sum = a[0]+a[1]+a[2];
      // std::cout<<sum<<std::endl;
     //  outs++;
//Extrae_event(60000000,0);
//       return sum;
/*    },
    [&](float a){
      //std::cout<<a<0std::endl;
      outs++;*/
    }
 );

  //std::cout<<outs<<std::endl;
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
    if(argv[3] == "l")
      e.set_queue_attributes(25,grppi::queue_mode::lockfree);
    else
      e.set_queue_attributes(25,grppi::queue_mode::blocking);
//    e.disable_ordering();
    equalizer(e,atoi(argv[2]),atoi(argv[3]));
  }
  if(string(argv[1]) == "seq"){
    auto e = grppi::sequential_execution{};
    equalizer(e,atoi(argv[2]),atoi(argv[3]));
  }
  if(string(argv[1]) == "omp"){
    auto e = grppi::parallel_execution_omp{};
    if(argv[3] == "l")
      e.set_queue_attributes(25,grppi::queue_mode::lockfree);
    else
      e.set_queue_attributes(25,grppi::queue_mode::blocking);
    equalizer(e,atoi(argv[2]),atoi(argv[3]));
  }
  return 0;
}
