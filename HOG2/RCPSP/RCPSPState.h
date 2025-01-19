//
// Created by idolu on 06/01/2025.
//
#pragma once

#include "petriclasses.h"
#include "readPetri.cpp"
#ifndef RCPSPSTATE_H
#define RCPSPSTATE_H
std::string finalstatename;
std::string initialstatename;
class RCPSPState {
  public:
  RCPSPState();
  RCPSPState(RCPSPState predecessor,Transition newTransition,bool status,int location,uint64_t &count);
  std::map<std::string, int> marking;
   std::vector<Transition> unstartedTransitions;
  std::vector<Transition> avilableTransition;
  std::vector<Transition> activeTransitions;
  //std::vector<RCPSPState> sons;
  //std::vector<int> unstartedTransitions;
  bool expanded=0;

  int name;
  int g=0;
  int h=0;

  //int GetG();
  //int checkEnd();
  bool operator==(const RCPSPState& other) const;
};


#endif // RCPSPSTATE_H