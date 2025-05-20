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
    ~RCPSPState() {
        // Clear STL containers explicitly (optional, as they would be destroyed automatically)
        marking.clear();
        unstartedTransitions.clear();
        avilableTransitionIndices.clear();
        activeTransitionIndices.clear();
        startedActivitiys.clear();
        finishedActivitiys.clear();

        // Any additional custom cleanup logic can go here
    }
  std::unordered_map<std::string, int> marking;
   std::vector<int> unstartedTransitions;
  //std::vector<Transition> avilableTransition;
  //std::vector<Transition> activeTransitions;
    std::vector<int> avilableTransitionIndices;  // Store transition IDs
    //std::vector<int> avilableDeTransitionIndices;  // Store transition IDs
    std::vector<std::pair<int, int>> activeTransitionIndices;  // Store transition ID and remaining duration

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

class RCPSPState_bi {
public:
    RCPSPState_bi();
    RCPSPState_bi(RCPSPState_bi predecessor,Transition newTransition,bool status,int location,uint64_t &count);
    ~RCPSPState_bi() {
        // Clear STL containers explicitly (optional, as they would be destroyed automatically)
        marking.clear();
        unstartedTransitions.clear();
        avilableTransitionIndices.clear();
        activeTransitionIndices.clear();
        startedActivitiys.clear();
        finishedActivitiys.clear();

        // Any additional custom cleanup logic can go here
    }
    std::unordered_map<std::string, int> marking;
    std::set<int> unstartedTransitions;
    std::vector<int> avilableTransitionIndices;  // Store transition IDs
    std::vector<int> avilableDeTransitionIndices;  // Store transition IDs
    std::vector<std::pair<int, int>> activeTransitionIndices;  // Store transition ID and remaining duration

    bool direction;
    bool nodestatus;
    double name=0;
    int predecesorname=0;

    std::set<int> startedActivitiys;
    std::set<int> finishedActivitiys;
    double g_f=0;
    double h_f=0;
    double g_b=0;
    double h_b=0;

    double f=0;


    //int GetG();
    //int checkEnd();
    bool operator==(const RCPSPState_bi& other) const;
};
class RCPSPState_TT {
public:
    // --- Constructors ---
    RCPSPState_TT();
    RCPSPState_TT(const RCPSPState_TT& predecessor, int newTransitionId, bool applyTransition, int location, uint64_t& count);
    ~RCPSPState_TT() {
        marking.clear();
        unstartedTransitions.clear();
        activeTransitionIndices.clear();
        startedActivitiys.clear();
        finishedActivitiys.clear();
    }

    // --- Resource state ---
    std::unordered_map<std::string, int> marking; // resourceName -> available amount

    // --- Transition states ---
    std::vector<int> unstartedTransitions;                         // transitions not yet started
    std::vector<std::pair<int, int>> activeTransitionIndices;     // (transitionID, remaining time)
    std::map<int, int> startedActivitiys;                          // activityID -> start time
    std::map<int, int> finishedActivitiys;                         // activityID -> finish time

    // --- Metadata for search ---
    bool direction = true;  // true = forward, false = backward
    bool nodestatus = false; // optional usage
    double name = 0;         // for debugging
    int predecesorname = 0;  // for debugging

    // --- Cost fields ---
    double g = 0;
    double h = 0;
    double f = 0;

    // --- Comparison ---
    bool operator==(const RCPSPState_TT& other) const;

    // --- Logic (to be implemented) ---
    std::vector<int> getAvailableTransitionIndices_TT() const;
    void startTransition(int transitionId, int currentTime);
    void advanceTime(int timeStep);
};




int computeEarlyFinishTime(int activityId);

#endif // RCPSPSTATE_H