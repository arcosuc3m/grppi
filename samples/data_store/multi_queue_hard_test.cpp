#include <iostream>
#include <vector>

#include "task_dist/multi_queue_hard.h"


int main ()
{
  std::set<int> conjunto2{10,50};

  grppi::multi_queue_hard<int,int> prueba{10};

  std::cout << "prueba is a multi_queue" << std::endl;

  prueba.registry(10);
  prueba.registry(30);
  prueba.registry(50);
  std::cout << "prueba register 10, 30, 50" << std::endl;

  std::cout << "prueba.empty() = " << prueba.empty() << std::endl;
  
  std::cout << "prueba.empty(10) = " << prueba.empty(10) << std::endl;
  
  prueba.push(10, 11);
  prueba.push(30, 31);
  prueba.push(50, 51);

  std::cout << "prueba push (10,11) - (30,31) - (50,51)" << std::endl;

  prueba.push(10, 12);
  prueba.push(30, 32);
  prueba.push(50, 52);

  std::cout << "prueba push (10,12) - (30,32) - (50,52)" << std::endl;

  prueba.push(10, 13);
  prueba.push(30, 33);
  prueba.push(50, 53);

  std::cout << "prueba push (10,13) - (30,33) - (50,53)" << std::endl;


  std::cout << "prueba.empty() = " << prueba.empty() << std::endl;

  std::cout << "prueba.empty(10) = " << prueba.empty(10) << std::endl;

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  for (int i=0; i<3; i++) {
    int num = prueba.pop(10);
    std::cout << "prueba pop(10) is " << num << std::endl;
  }

  
  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  for (int i=0; i<3; i++) {
    int num = prueba.pop();
    std::cout << "prueba pop is " << num << std::endl;
  }

  while (! prueba.empty(50)) {
    int num = prueba.pop(50);
    std::cout << "prueba pop (50)  is " << num << std::endl;
  }

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }
  prueba.push(30, 34);
  std::cout << "prueba push (30,34)" << std::endl;

  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }

  for (int i=10; i<20; i++) {
    prueba.push(10, i);
    std::cout << "prueba push (10, "<< i << ") " << std::endl;
    prueba.empty();
  }
  for (int i=10; i<20; i++) {
    int num = prueba.pop(10);
    std::cout << "prueba pop (10) is " << num << std::endl;
    prueba.empty();
  }

  for (int i=10; i<20; i++) {
    prueba.push(10, i);
    std::cout << "prueba push (10, "<< i << ") " << std::endl;
  }

  prueba.empty();
  prueba.push(30, 31);
  std::cout << "prueba push (30, 31) " << std::endl;

  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }

  std::cout << "prueba CLEAR ALL LABELS"<< std::endl;
  {std::vector<int> aux_labels = {10,30,50};
  for (auto const& lab : aux_labels) {
    while (! prueba.empty(lab)) {
      int num = prueba.pop(lab);
      std::cout << "prueba pop(" << lab << ")  is " << num << std::endl;
    }
  }}
  
  prueba.push(std::vector<int>{10}, 11, false);
  prueba.push(std::vector<int>{30}, 31, true);
  prueba.push(std::vector<int>{50}, 51, false);

  std::cout << "prueba push (10,11,false) - (30,31,true) - (50,51,false)" << std::endl;

  prueba.push(std::vector<int>{10}, 12, true);
  prueba.push(std::vector<int>{30}, 32, false);
  prueba.push(std::vector<int>{50}, 52, true);

  std::cout << "prueba push (10,12,true) - (30,32,false) - (50,52,true)" << std::endl;

  prueba.push(std::vector<int>{10}, 13, false);
  prueba.push(std::vector<int>{30}, 33, true);
  prueba.push(std::vector<int>{50}, 53, false);
  
  std::cout << "prueba push (10,13,false) - (30,33,true) - (50,53,false)" << std::endl;

  std::cout << "prueba.empty() = " << prueba.empty() << std::endl;

  std::cout << "prueba.empty(10) = " << prueba.empty(10) << std::endl;

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  for (int i=0; i<3; i++) {
    int num = prueba.pop(10);
    std::cout << "prueba pop(10) is " << num << std::endl;
  }

  
  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  for (int i=0; i<3; i++) {
    int num = prueba.pop();
    std::cout << "prueba pop is " << num << std::endl;
  }

  while (! prueba.empty(50)) {
    int num = prueba.pop(50);
    std::cout << "prueba pop (50)  is " << num << std::endl;
  }

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }
  std::cout << "prueba.empty() = " << prueba.empty() << std::endl;

  std::cout << "prueba.empty(10) = " << prueba.empty(10) << std::endl;

  std::cout << "prueba.empty(30) = " << prueba.empty(30) << std::endl;

  std::cout << "prueba.empty(50) = " << prueba.empty(50) << std::endl;

  std::cout << "prueba.empty_all() = " << prueba.empty_all() << std::endl;

  std::cout << "prueba.count() = " << prueba.count() << std::endl;


  prueba.push(std::vector<int>{50}, 52, true);

  std::cout << "prueba push (50,52,true) " << std::endl;

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }

  std::cout << "prueba.empty() = " << prueba.empty() << std::endl;

  std::cout << "prueba.empty(10) = " << prueba.empty(10) << std::endl;

  std::cout << "prueba.empty(30) = " << prueba.empty(30) << std::endl;

  std::cout << "prueba.empty(50) = " << prueba.empty(50) << std::endl;

  std::cout << "prueba.empty_all() = " << prueba.empty_all() << std::endl;

  std::cout << "prueba.count() = " << prueba.count() << std::endl;

  std::cout << "prueba CLEAR ALL LABELS"<< std::endl;
  {std::vector<int> aux_labels = {10,30,50};
  for (auto const& lab : aux_labels) {
    while (! prueba.empty(lab)) {
      int num = prueba.pop(lab);
      std::cout << "prueba pop(" << lab << ")  is " << num << std::endl;
    }
  }}
  
  prueba.push(std::vector<int>{10}, 11, false);
  prueba.push(std::vector<int>{30}, 31, true);
  prueba.push(std::vector<int>{10,50}, 51, false);

  std::cout << "prueba push (10,11,false) - (30,31,true) - (10-50,51,false)" << std::endl;

  prueba.push(std::vector<int>{10}, 12, true);
  prueba.push(std::vector<int>{30}, 32, false);
  prueba.push(std::vector<int>{10,50}, 52, true);

  std::cout << "prueba push (10,12,true) - (30,32,false) - (10-50,52,true)" << std::endl;

  prueba.push(std::vector<int>{10}, 13, false);
  prueba.push(std::vector<int>{30}, 33, true);
  prueba.push(std::vector<int>{10,50}, 53, false);

  std::cout << "prueba push (10,13,false) - (30,33,true) - (10-50,53,false)" << std::endl;

  std::cout << "prueba.empty() = " << prueba.empty() << std::endl;

  std::cout << "prueba.empty(10) = " << prueba.empty(10) << std::endl;

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  for (int i=0; i<3; i++) {
    int num = prueba.pop(10);
    std::cout << "prueba pop(10) is " << num << std::endl;
  }

  
  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  for (int i=0; i<3; i++) {
    int num = prueba.pop();
    std::cout << "prueba pop is " << num << std::endl;
  }

  while (! prueba.empty(50)) {
    int num = prueba.pop(50);
    std::cout << "prueba pop (50)  is " << num << std::endl;
  }

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  
  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }
  std::cout << "prueba.empty() = " << prueba.empty() << std::endl;

  std::cout << "prueba.empty(10) = " << prueba.empty(10) << std::endl;

  std::cout << "prueba.empty(30) = " << prueba.empty(30) << std::endl;

  std::cout << "prueba.empty(50) = " << prueba.empty(50) << std::endl;

  std::cout << "prueba.empty_all() = " << prueba.empty_all() << std::endl;

  std::cout << "prueba.count() = " << prueba.count() << std::endl;


  prueba.push(std::vector<int>{50}, 54, true);

  std::cout << "prueba push (50,54,true) " << std::endl;

  {
    std::cout << "prueba.loaded_set {10, 50}" << std::endl;
    auto conjunto = prueba.loaded_set(conjunto2);
    for (int i=0; i<60; i=i+10) {
        if (conjunto.find(i) != conjunto.end()) {
            std::cout << "prueba.loaded_set " << i << " is found" << std::endl;
        }
    }
  }
  while (! prueba.empty()) {
    int num = prueba.pop();
    std::cout << "prueba pop  is " << num << std::endl;
  }

  std::cout << "prueba.empty() = " << prueba.empty() << std::endl;

  std::cout << "prueba.empty(10) = " << prueba.empty(10) << std::endl;

  std::cout << "prueba.empty(30) = " << prueba.empty(30) << std::endl;

  std::cout << "prueba.empty(50) = " << prueba.empty(50) << std::endl;

  std::cout << "prueba.empty_all() = " << prueba.empty_all() << std::endl;

  std::cout << "prueba.count() = " << prueba.count() << std::endl;

  std::cout << "prueba CLEAR ALL LABELS"<< std::endl;
  {std::vector<int> aux_labels = {10,30,50};
  for (auto const& lab : aux_labels) {
    while (! prueba.empty(lab)) {
      int num = prueba.pop(lab);
      std::cout << "prueba pop(" << lab << ")  is " << num << std::endl;
    }
  }}
  
  return 0;
}
