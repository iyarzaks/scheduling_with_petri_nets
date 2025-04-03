//
// Created by idolu on 29/12/2024.
//
#ifndef RCPSP_H
#define RCPSP_H
//#include "../algorithms/OldSearchEnvironment.h"
#include "../search/SearchEnvironment.h"
#include "RCPSPState.h"
#include "../utils//GLUtil.h"
#include <functional>


//creted the RCPSPState in searchgraph
// class RCPSPState{
// searchNode node;
//   };
std::uint64_t count=0;
//int Nsize=0;
typedef int action;

class RCPSP : public SearchEnvironment<RCPSPState,int>{
  public:
  RCPSP();
  void GetSuccessors(const RCPSPState &nodeID, std::vector<RCPSPState> &neighbors) const override;
  bool GoalTest(const RCPSPState &node, const RCPSPState &goal) const override;
	double HCost(const RCPSPState &state1, const RCPSPState &state2) const override;
	double GCost(const RCPSPState &state1, const RCPSPState &state2) const override;

  int GetAction(const RCPSPState &nodeID, const RCPSPState &nodeID2) const override;
  int GetNumSuccessors(const RCPSPState &stateID) const;
  void GetActions(const RCPSPState &nodeID, std::vector<int> &actions) const override;
  void ApplyAction(RCPSPState &s, int a) const override;
  uint64_t GetActionHash(int act) const;
  uint64_t GetStateHash(const RCPSPState &node) const;
  bool InvertAction(int &a) const;
  std::vector<RCPSPState> GetSuccessors(const RCPSPState &nodeID) const;
  double GCost(const RCPSPState &node, const int &act) const override;
  };
std::chrono::duration<double> hashTIME;
std::chrono::duration<double> secssesorTIME;






inline uint64_t RCPSP::GetStateHash(const RCPSPState &node) const {
  auto startS1 = std::chrono::high_resolution_clock::now();

  std::size_t seed = 0;

  for (const auto& pair : node.startedActivitiys) {
    seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  for (const auto& pair : node.finishedActivitiys) {
    seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  auto endS1 = std::chrono::high_resolution_clock::now();

  hashTIME += endS1-startS1;
  return seed;

}


inline RCPSP::RCPSP() {
}

inline void RCPSP::GetSuccessors(const RCPSPState &nodeID, std::vector<RCPSPState> &neighbors) const {
  if (nodeID.activeTransitions.size()>0) {

    count++;
    int t=0;
    for (int i=0;i<nodeID.activeTransitions.size();i++) {
      if (nodeID.activeTransitions[i].duration<nodeID.activeTransitions[t].duration) {
        t=i;
      }
    }
    neighbors.emplace_back(RCPSPState(nodeID,nodeID.activeTransitions[t],0,t,count));
  }

  for (int i=0;i<nodeID.avilableTransition.size();i++) {
    count++;
    neighbors.emplace_back(RCPSPState(nodeID,nodeID.avilableTransition[i],1,i,count));
  }
  auto endS1 = std::chrono::high_resolution_clock::now();

}

inline bool RCPSP::GoalTest(const RCPSPState &node, const RCPSPState &goal) const {
  if (node.marking.at(finalstatename) == 1) {
  return true;
}
  return false;
}

inline double RCPSP::HCost(const RCPSPState &state1, const RCPSPState &state2) const {
  return state1.h-state2.h;
return state1.h;
}

inline double RCPSP::GCost(const RCPSPState &state1, const RCPSPState &state2) const {
  // int remain=0;
  // for (int i = state2.activeTransitions.size() - 1; i >= 0; --i) {
  //   if (state2.activeTransitions[i].duration > remain) {
  //     remain = state2.activeTransitions[i].duration;
  //   }
  // }
    //return remain-state2.g;
   return state2.g-state1.g;//+state1.g
    //return state2.g;

}
//NOT IN USE OF A*
inline uint64_t RCPSP::GetActionHash(int act) const {
  // Example hash for an action
  return std::hash<int>()(act);
}
inline void RCPSP::GetActions(const RCPSPState &nodeID, std::vector<int> &actions) const {
  // for (int i = 0; i < nodeID.sons.size(); ++i) {
  //   actions.push_back(i); // Add the index of each available transition as an action.
  // }
}

inline bool RCPSP::InvertAction(int &a) const {
  // Example logic to invert an action
  return true;
}

inline std::vector<RCPSPState> RCPSP::GetSuccessors(const RCPSPState &nodeID) const {

     std::vector<RCPSPState> neighbors;
    // for (int i = 0; i < nodeID.sons.size(); ++i) {
    //   neighbors.push_back(nodeID.sons[i]);
    // }
    return neighbors;
}


inline int RCPSP::GetAction(const RCPSPState &nodeID, const RCPSPState &nodeID2) const {
}

//inline uint64_t RCPSP::GetStateHash(const RCPSPState &s) const {
 // return s.name;
//}
inline int RCPSP::GetNumSuccessors(const RCPSPState &stateID) const {
  int i=0;
  if (stateID.activeTransitions.size() > 0){i=1;}
  return stateID.avilableTransition.size()+i;
}



inline void RCPSP::ApplyAction(RCPSPState &s, int a) const {

}
inline double RCPSP::GCost(const RCPSPState &node, const int &act) const {
  return node.g;
}
class RCPSP_BiGreedy : public SearchEnvironment<RCPSPState, action> {
public:

  inline void GetSuccessors(const RCPSPState &nodeID, std::vector<RCPSPState> &neighbors) const override {
    if (nodeID.activeTransitions.size() > 0) {
     // count++;
      int t = 0;
      for (int i = 0; i < nodeID.activeTransitions.size(); i++) {
        if (nodeID.activeTransitions[i].duration < nodeID.activeTransitions[t].duration) {
          t = i;
        }
      }
      neighbors.emplace_back(RCPSPState(nodeID, nodeID.activeTransitions[t], 0, t, count));
    }

    for (int i = 0; i < nodeID.avilableTransition.size(); i++) {
     // count++;
      neighbors.emplace_back(RCPSPState(nodeID, nodeID.avilableTransition[i], 1, i, count));
    }
  }

  inline bool GoalTest(const RCPSPState &node, const RCPSPState &goal) const override {
  if (goal.marking.at(finalstatename) == 1) {
    if (node.name == 0) {
      return false;
    }
    return node.marking.at(finalstatename) == 1;
  }
  //else {
   // return node.marking.at("_pre_1") == 1;
  //}
  }

  inline double HCost(const RCPSPState &state1, const RCPSPState &state2) const override {
    return state1.g + state1.h; // Trick BidirectionalGreedyBestFirst into behaving like A*
  }

  inline double GCost(const RCPSPState &state1, const RCPSPState &state2) const override {
    return state2.g - state1.g; // Track actual transition cost
  }
  inline void GetActions(const RCPSPState &state, std::vector<action> &actions) const override {
    // Not used in BidirectionalGreedyBestFirst, but must be implemented
    return;
  }
  virtual action GetAction(const RCPSPState &state1, const RCPSPState &state2) const override {
    // // Determine what action takes you from state1 to state2
    // // For RCPSP, this might be the index of the transition or activity that was started/completed
    //

    // // Compare the states to figure out what changed
    // for (int i = 0; i < state2.activeTransitions.size(); i++) {
    //   bool foundInState1 = false;
    //   for (int j = 0; j < state1.activeTransitions.size(); j++) {
    //     if (state2.activeTransitions[i].name == state1.activeTransitions[j].name) {
    //       foundInState1 = true;
    //       break;
    //     }
    //   }
    //   if (!foundInState1) {
    //     // This activity was started between state1 and state2
    //     return static_cast<action>(state2.activeTransitions[i].name);
    //   }
    // }
    //
    // // Otherwise, check if an activity was completed
    // for (int i = 0; i < state1.activeTransitions.size(); i++) {
    //   bool foundInState2 = false;
    //   for (int j = 0; j < state2.activeTransitions.size(); j++) {
    //     if (state1.activeTransitions[i].name == state2.activeTransitions[j].name) {
    //       foundInState2 = true;
    //       break;
    //     }
    //   }
    //   if (!foundInState2) {
    //     // This activity was completed between state1 and state2
    //     return static_cast<action>(-state1.activeTransitions[i].name);  // Negative to indicate completion
    //   }
    // }
    //
    // // If we can't determine the action, return a default
    return static_cast<action>(0);
  }
  inline void ApplyAction(RCPSPState &state, action action) const override {
    // Not used, but required for abstract class
  }

  inline void UndoAction(RCPSPState &state, action action) const override {
    // Not needed for bidirectional search, but required
  }
double GCost(const RCPSPState &node, const action &act) const override {
    return node.g;
  };
  bool InvertAction(action& a) const override {
    return false; // Replace with appropriate logic
  }
  uint64_t GetActionHash(action act) const override {
    return 0;
  };
  uint64_t GetStateHash(const RCPSPState &node) const override {
    auto startS1 = std::chrono::high_resolution_clock::now();

    std::size_t seed = 0;

    for (const auto& pair : node.startedActivitiys) {
      seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    for (const auto& pair : node.finishedActivitiys) {
      seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    auto endS1 = std::chrono::high_resolution_clock::now();

    hashTIME += endS1-startS1;
    return seed;

  };
};


#endif //RCPSP_H
//
// Created by idolu on 06/01/2025.
//
