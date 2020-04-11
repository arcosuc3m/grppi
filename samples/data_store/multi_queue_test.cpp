#include <iostream>

#include "task_dist/multi_queue.h"


int main ()
{
  std::set<int> conjunto2{10,50};

  grppi::multi_queue<int,int> prueba(10);
  
  std::cout << "prueba is a multi_queue" << std::endl;

  prueba.registry(10);
  prueba.registry(30);
  prueba.registry(50);

  if (prueba.empty()) {
    std::cout << "prueba is empty" << std::endl;
  }
  if (prueba.empty(10)) {
    std::cout << "prueba 10 is empty" << std::endl;
  }
  
  prueba.push(10, 11);
  prueba.push(30, 31);
  prueba.push(50, 51);
  
  prueba.push(10, 12);
  prueba.push(30, 32);
  prueba.push(50, 52);

  prueba.push(10, 13);
  prueba.push(30, 33);
  prueba.push(50, 53);

  if (prueba.empty()) {
    std::cout << "prueba is empty" << std::endl;
  }
  if (prueba.empty(10)) {
    std::cout << "prueba 10 is empty" << std::endl;
  }

  {
    std::cout << "prueba.loaded_set 1" << std::endl;
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
    std::cout << "prueba.loaded_set 2" << std::endl;
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
    std::cout << "prueba.loaded_set 3" << std::endl;
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
    std::cout << "prueba.loaded_set 4" << std::endl;
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

  return 0;
}
