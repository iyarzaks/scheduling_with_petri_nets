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
  std::unordered_map<std::string, int> marking;
   std::vector<int> unstartedTransitions;
  std::vector<Transition> avilableTransition;
  std::vector<Transition> activeTransitions;
    bool direction;
    bool nodestatus;
  //std::vector<RCPSPState> sons;
  //std::vector<int> unstartedTransitions;
  double name=0;
  int predecesorname=0;

  std::map<int, int> startedActivitiys;
  std::map<int, int> finishedActivitiys;
  double g=0;
  double h=0;

  //int GetG();
  //int checkEnd();
  bool operator==(const RCPSPState& other) const;
};
int computeEarlyFinishTime(int activityId);

#endif // RCPSPSTATE_H