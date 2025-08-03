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
std::vector<std::string> resourceNames = {"R1", "R2", "R3", "R4"};

class RCPSPState_TT {
public:
    // --- Constructors ---
    RCPSPState_TT();
   // RCPSPState_TT(const RCPSPState_TT& predecessor, int newTransitionId, bool applyTransition, int location, uint64_t& count);
    RCPSPState_TT(const RCPSPState_TT& prev, int ID, int firingTime);
    ~RCPSPState_TT() {
        marking.clear();
        unstartedTransitions.clear();
        startedActivitiys.clear();
        finishedActivitiys.clear();
    }

    // --- Resource state ---
    std::unordered_map<std::string, std::vector<std::pair<int, int>>> marking;
    std::vector<int> unstartedTransitions;                         // transitions not yet started
    std::map<int, int> startedActivitiys;                          // activityID -> start time
    std::map<int, int> finishedActivitiys;                         // activityID -> finish time
    std::vector<std::pair<int, int>> avilableTransitionIndices;  // Store transition IDs

    double g = 0;
    double h = 0;

    // --- Comparison ---
    bool operator==(const RCPSPState_TT& other) const;

    // --- Logic (to be implemented) ---
  //  std::vector<int> getAvailableTransitionIndices_TT() const;
    // std::vector<std::pair<int, int>> getAvailableTransitionIndices_TT(
    //     const std::vector<int> &unstartedTransitions,
    //     const std::map<int, int> &finishedActivities,
    //     const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &marking) ;
//     std::vector<std::pair<int, int>> checkAvailableTransitions_TT(
//         const std::vector<int> &optionalTransitions,
//         const std::map<int, int> &finishedActivities,
//         const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &marking
//     );
//     std::vector<int> getOptionalTransitions_TT(
//     const std::vector<int> &unstartedTransitions,
//     const std::map<int, int> &startedActivities  // Note: started, not finished
// ) ;
    // void startTransition(int transitionId, int currentTime);

};




int computeEarlyFinishTime(int activityId);

#endif // RCPSPSTATE_H