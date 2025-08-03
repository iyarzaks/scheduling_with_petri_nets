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





inline uint64_t RCPSP::GetStateHash(const RCPSPState &node) const {
  //auto startS1 = std::chrono::high_resolution_clock::now();

  std::size_t seed = 0;

  for (const auto& pair : node.startedActivitiys) {
    seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  for (const auto& pair : node.finishedActivitiys) {
    seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  //auto endS1 = std::chrono::high_resolution_clock::now();

  //hashTIME += endS1-startS1;
  return seed;

}


inline RCPSP::RCPSP() {
}
/*
inline void RCPSP::GetSuccessors(const RCPSPState &nodeID, std::vector<RCPSPState> &neighbors) const {


  if (nodeID.activeTransitions.size()>0) {

    count++;
    int t=0;
    for (int i=0;i<nodeID.activeTransitions.size();i++) {
       if (nodeID.activeTransitions[i].duration<nodeID.activeTransitions[t].duration) {
         t=i;
       }
      //neighbors.emplace_back(RCPSPState(nodeID,nodeID.activeTransitions[t],0,t,count));
    }
    neighbors.emplace_back(RCPSPState(nodeID,nodeID.activeTransitions[t],0,t,count));
  }
  for (int i=0;i<nodeID.avilableTransition.size();i++) {
    count++;
    neighbors.emplace_back(RCPSPState(nodeID,nodeID.avilableTransition[i],1,i,count));
  }
  auto endS1 = std::chrono::high_resolution_clock::now();

}
*/
inline void RCPSP::GetSuccessors(const RCPSPState &nodeID, std::vector<RCPSPState> &neighbors) const {
  // Handle active transitions
  if (!nodeID.activeTransitionIndices.empty()) {
    count++;
    int t = 0;
    // Find the transition with the minimum remaining duration
    for (int i = 0; i < nodeID.activeTransitionIndices.size(); i++) {
      if (nodeID.activeTransitionIndices[i].second < nodeID.activeTransitionIndices[t].second) {
        t = i;
      }
    }

    // Get the actual transition object using the index
    int transitionIdx = nodeID.activeTransitionIndices[t].first;
    Transition transition = petri.Transitions[transitionIdx - 1];
    transition.duration= nodeID.activeTransitionIndices[t].second;

    // Create successor state
    neighbors.emplace_back(RCPSPState(nodeID, transition, false, t, count));
  }

  // Handle available transitions
  for (int i = 0; i < nodeID.avilableTransitionIndices.size(); i++) {
    count++;
    // Get the actual transition object using the index
    int transitionIdx = nodeID.avilableTransitionIndices[i];
    const Transition& transition = petri.Transitions[transitionIdx - 1];

    // Create successor state
    neighbors.emplace_back(RCPSPState(nodeID, transition, true, i, count));
  }

}

inline bool RCPSP::GoalTest(const RCPSPState &node, const RCPSPState &goal) const {
  if (node.marking.at(finalstatename) == 1) {
  return true;
}
  return false;
}
/*
inline double calculateEarlyFinishRecursive(int activityId, std::map<int, int>& earlyfinishMap,
                                            const std::vector<int>& unstartedTransitions,
                                            const std::vector<Transition>& activeTransitions,
                                            const RCPSP_example& RCPSPex) {
  // If we've already computed this activity's early finish time, return it
  if (earlyfinishMap.find(activityId) != earlyfinishMap.end()) {
    return earlyfinishMap[activityId];
  }

  int maxFinishTime = 0;

  // Process all dependencies
  for (const auto &dep: RCPSPex.backword_dependencies[activityId - 1]) {
    int depId = std::stoi(dep) - 1;

    // Recursively compute the early finish time of the dependency if not already computed
    if (earlyfinishMap.find(depId + 1) == earlyfinishMap.end()) {
      calculateEarlyFinishRecursive(depId + 1, earlyfinishMap, unstartedTransitions, activeTransitions, RCPSPex);
    }

    if (std::find(unstartedTransitions.begin(), unstartedTransitions.end(), depId + 1) != unstartedTransitions.end()) {
      int duration = getTransitionDuration(activeTransitions, std::stoi(dep));
      if (duration != -1) {
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId + 1] + duration);
      } else {
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId + 1] + RCPSPex.activities[depId].duration);
      }
    } else {
      maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId + 1]);
    }
  }

  // Store and return the result
  earlyfinishMap[activityId] = maxFinishTime;
  return maxFinishTime;
}
*/

double calculateEarlyFinishRecursive(int activityId, std::map<int, int>& earlyfinishMap,
                                    const std::vector<int>& unstartedTransitions,
                                    const std::vector<std::pair<int, int>>& activeTransitions,
                                    const RCPSP_example& RCPSPex) {
  // If we've already computed this activity's early finish time, return it
  if (earlyfinishMap.find(activityId) != earlyfinishMap.end()) {
    return earlyfinishMap[activityId];
  }

  int maxFinishTime = 0;

  // Process all dependencies
  for (const auto &dep: RCPSPex.backword_dependencies[activityId - 1]) {
    int depId = std::stoi(dep) - 1;

    // Recursively compute the early finish time of the dependency if not already computed
    if (earlyfinishMap.find(depId + 1) == earlyfinishMap.end()) {
      calculateEarlyFinishRecursive(depId + 1, earlyfinishMap, unstartedTransitions, activeTransitions, RCPSPex);
    }

    if (std::find(unstartedTransitions.begin(), unstartedTransitions.end(), depId + 1) != unstartedTransitions.end()) {
      int duration = getTransitionDuration2(activeTransitions, std::stoi(dep));
      if (duration != -1) {
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId + 1] + duration);
      } else {
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId + 1] + RCPSPex.activities[depId].duration);
      }
    } else {
      maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId + 1]);
    }
  }

  // Store and return the result
  earlyfinishMap[activityId] = maxFinishTime;
  return maxFinishTime;
}




inline double RCPSP::HCost(const RCPSPState &state1, const RCPSPState &state2) const {
 return state1.h;
  // for (int activityId: state1.unstartedTransitions) {
  //   int maxFinishTime = 0;
  //   std::set<int> processedDependencies;
  //
  //   for (const auto &dep: RCPSPex.backword_dependencies[activityId - 1]) {
  //     int depId = std::stoi(dep) - 1;
  //     // if (processedDependencies.count(depId) > 0) continue;
  //     // processedDependencies.insert(depId);
  //     if (std::find(state1.unstartedTransitions.begin(), state1.unstartedTransitions.end(), depId + 1) != state1.unstartedTransitions.end()) {
  //       int duration = getTransitionDuration2(state1.activeTransitionIndices, std::stoi(dep));
  //       if (duration !=-1) {
  //         maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1] + duration);
  //         //if (RCPSPex.activities[depId].duration !=duration) {
  //         //  std::cout<<name<<":"<<dep<<" "<<activityId<<" "<<RCPSPex.activities[depId].duration-duration<<std::endl;
  //         //}
  //       }
  //       else {
  //         maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1] + RCPSPex.activities[depId].duration);
  //
  //       }
  //     }
  //     else {
  //       maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1]);
  //     }
  //   }
  //
  //   earlyfinishMap2[activityId] = maxFinishTime;
  //   //std::cout <<activityId<<":"<< earlyfinishMap[activityId]+RCPSPex.activities[activityId-1].duration << std::endl;
  //   // For last element with duration 0, just use the max finish time of dependencies
  // }
  // if (earlyfinishMap2.size()==0) {
  //   h = 0;
  // }
  // else {
  //   h = earlyfinishMap2.rbegin()->second;;
  //
  // }
  // // if (h != newH) {
  // //   int asd;
  // //   asd++;
  // // }
  // auto endS3 = std::chrono::high_resolution_clock::now();
  // HTIME += endS3 - startS3;
  //
  // return h;
  /* double newH = 0;
  // std::map<int, int> earlyfinishMap; // Map to store activity IDs and their early finish times
  //
  // if (!state1.unstartedTransitions.empty()) {
  //   // We want the maximum early finish time, which should be the last activity
  //   int lastActivity = state1.unstartedTransitions.back(); // Assuming the last activity is the one we care about
  //   newH = calculateEarlyFinishRecursive(lastActivity, earlyfinishMap,
  //                                   state1.unstartedTransitions,
  //                                   state1.activeTransitions,
  //                                   RCPSPex);
  // }
  // auto endS3 = std::chrono::high_resolution_clock::now();
  // HTIME += endS3 - startS3;
  // return newH;*/

}
/*
inline double RCPSP::HCost(const RCPSPState &state1, const RCPSPState &state2) const {
  auto startS3 = std::chrono::high_resolution_clock::now();

  double newH = 0;
  std::map<int, int> earlyfinishMap; // Map to store activity IDs and their early finish times

  if (!state1.unstartedTransitions.empty()) {
    // Convert activeTransitionIndices to a compatible format for the recursive function
    std::vector<std::pair<int, int>> activeTransitions;
    for (const auto& [transIdx, duration] : state1.activeTransitionIndices) {
      activeTransitions.push_back({transIdx, duration});
    }

    // We want the maximum early finish time, which should be the last activity
    int lastActivity = state1.unstartedTransitions.back(); // Assuming the last activity is the one we care about
    newH = calculateEarlyFinishRecursive(lastActivity, earlyfinishMap,
                                    state1.unstartedTransitions,
                                    activeTransitions,
                                    RCPSPex);
  }

  auto endS3 = std::chrono::high_resolution_clock::now();
  HTIME += endS3 - startS3;
  return newH;
}
*/

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
  return 0;
}

//inline uint64_t RCPSP::GetStateHash(const RCPSPState &s) const {
 // return s.name;
//}
inline int RCPSP::GetNumSuccessors(const RCPSPState &stateID) const {
  // int i=0;
  // if (stateID.activeTransitions.size() > 0){i=1;}
  // return stateID.avilableTransition.size()+i;
  return 0;
}



inline void RCPSP::ApplyAction(RCPSPState &s, int a) const {

}
inline double RCPSP::GCost(const RCPSPState &node, const int &act) const {
  return node.g;
}

class RCPSP_BiGreedy : public SearchEnvironment<RCPSPState_bi, action> {
public:

inline void GetSuccessors(const RCPSPState_bi &nodeID, std::vector<RCPSPState_bi> &neighbors) const override {
neighbors.clear();
  if (GetExpandForward) {
    if (!nodeID.activeTransitionIndices.empty()) {
      count++;
      int t = 0;
      // Find the transition with the minimum remaining duration
      for (int i = 0; i < nodeID.activeTransitionIndices.size(); i++) {
        if (nodeID.activeTransitionIndices[i].second < nodeID.activeTransitionIndices[t].second) {
          t = i;
        }
      }

      // Get the actual transition object using the index
      int transitionIdx = nodeID.activeTransitionIndices[t].first;
      Transition transition = petri.Transitions[transitionIdx - 1];
      transition.duration= nodeID.activeTransitionIndices[t].second;

      // Create successor state
      neighbors.emplace_back(RCPSPState_bi(nodeID, transition, false, t, count));
    }

    // Handle available transitions
    for (int i = 0; i < nodeID.avilableTransitionIndices.size(); i++) {
      count++;
      // Get the actual transition object using the index
      int transitionIdx = nodeID.avilableTransitionIndices[i];
      const Transition& transition = petri.Transitions[transitionIdx - 1];

      // Create successor state
      neighbors.emplace_back(RCPSPState_bi(nodeID, transition, true, i, count));
    }
  }
else {
      if (nodeID.activeTransitionIndices.size() > 0) {
        count++;
        int t = 0;
        for (int i = 0; i < nodeID.activeTransitionIndices.size(); i++) {
           if (petri.Transitions[nodeID.activeTransitionIndices[i].first-1].duration -nodeID.activeTransitionIndices[i].second< petri.Transitions[nodeID.activeTransitionIndices[t].first-1].duration -nodeID.activeTransitionIndices[t].second) {
            t = i;
          }
        }
        int transitionIdx = nodeID.activeTransitionIndices[t].first;
        Transition transition = petri.Transitions[transitionIdx - 1];
        transition.duration= nodeID.activeTransitionIndices[t].second;

        // Create successor state
        neighbors.emplace_back(RCPSPState_bi(nodeID, transition, false, t, count));      }

      for (int i = 0; i < nodeID.avilableDeTransitionIndices.size(); i++) {
        count++;
        int transitionIdx = nodeID.avilableDeTransitionIndices[i];
        const Transition& transition = petri.Transitions[transitionIdx - 1];

        // Create successor state
        neighbors.emplace_back(RCPSPState_bi(nodeID, transition, true, i, count));      }


    }
  }
   inline bool GoalTest(const RCPSPState_bi &node, const RCPSPState_bi &goal) const override {
return false;
    if (goal.marking.at(finalstatename) == 1) {
     if (node.name == 0) {
       return false;
     }
     return node.marking.at(finalstatename) == 1;
   }
  else {
    return node.marking.at("_pre_1") == 1;
  }
  }

  inline double HCost(const RCPSPState_bi &state1, const RCPSPState_bi &state2) const override {
  std::map<int, int> earlyfinishMap; // Map to store activity IDs and their early finish times
    //std::map<int, int> visitmap; // Map to store activity IDs and their early finish times
    double h;
    std::set<int> processedDependencies;
    // Iterate over unstarted activitiesint lastElementEarlyFinish = 0;
    //int lastElementEarlyFinish = 0;
    for (int activityId: state1.unstartedTransitions) {
      int maxFinishTime = 0;
      std::set<int> processedDependencies;

      for (const auto &dep: RCPSPex.backword_dependencies[activityId - 1]) {
        int depId = std::stoi(dep) - 1;
        // if (processedDependencies.count(depId) > 0) continue;
        // processedDependencies.insert(depId);
        if (std::find(state1.unstartedTransitions.begin(), state1.unstartedTransitions.end(), depId + 1) != state1.unstartedTransitions.end()) {
          //maby state1 or state2
          int duration = getTransitionDuration2(state1.activeTransitionIndices, std::stoi(dep));
          if (duration !=-1) {
            maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId+1] + duration);
            //if (RCPSPex.activities[depId].duration !=duration) {
            //  std::cout<<name<<":"<<dep<<" "<<activityId<<" "<<RCPSPex.activities[depId].duration-duration<<std::endl;
            //}
          }
          else {
            maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId+1] + RCPSPex.activities[depId].duration);

          }
        }
        else {
          maxFinishTime = std::max(maxFinishTime, earlyfinishMap[depId+1]);
        }
      }

      earlyfinishMap[activityId] = maxFinishTime;
      //std::cout <<activityId<<":"<< earlyfinishMap[activityId]+RCPSPex.activities[activityId-1].duration << std::endl;
      // For last element with duration 0, just use the max finish time of dependencies
    }
    if (earlyfinishMap.size()==0) {
      h = 0;
    }
    else {
      h = earlyfinishMap.rbegin()->second;;

    }
    return h;
  }

  //
   inline double GCost(const RCPSPState_bi &state1, const RCPSPState_bi &state2) const override {
  return 0;
  if (state2.direction) {
  return state2.g_f - state1.g_f;
}
else {
    return state2.g_b - state1.g_b;
  }
  // Track actual transition cost
   }
  inline void GetActions(const RCPSPState_bi &state, std::vector<action> &actions) const override {
    // Not used in BidirectionalGreedyBestFirst, but must be implemented
    return;
  }
  virtual action GetAction(const RCPSPState_bi &state1, const RCPSPState_bi &state2) const override {

    return static_cast<action>(0);
  }
  inline void ApplyAction(RCPSPState_bi &state, action action) const override {
    // Not used, but required for abstract class
  }

  inline void UndoAction(RCPSPState_bi &state, action action) const override {
    // Not needed for bidirectional search, but required
  }
double GCost(const RCPSPState_bi &node, const action &act) const override {
if (node.direction) {
    return node.g_f;
}
else {
  return node.g_b;
}
  };
  bool InvertAction(action& a) const override {
    return false; // Replace with appropriate logic
  }
  uint64_t GetActionHash(action act) const override {
    return 0;
  };

  uint64_t GetStateHash(const RCPSPState_bi &node) const {
    //auto startS1 = std::chrono::high_resolution_clock::now();
    std::size_t seed = 0;
    // Hash the marking (Petri net state)
    for (const auto& pair : node.activeTransitionIndices) {
      seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    // Hash the startedActivitiys vector
    for (const auto& activity : node.startedActivitiys) {
      seed ^= std::hash<int>{}(activity) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    // Hash the finishedActivitiys vector
    for (const auto& activity : node.finishedActivitiys) {
      seed ^= std::hash<int>{}(activity) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    //auto endS1 = std::chrono::high_resolution_clock::now();
    //hashTIME += endS1 - startS1;

    return seed;

    // Hash the finished activities
    // for (const auto& pair : node.finishedActivitiys) {
    //   seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    //   seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    // }

    // Distinguish forward vs. backward search by modifying the seed
    //seed ^= (GetExpandForward ? 0xAAAAAAAAAAAAAAAA : 0x5555555555555555);


  }

};
class ForwardRCPSPHeuristic : public Heuristic<RCPSPState_bi> {
public:
  double HCost(const RCPSPState_bi &current, const RCPSPState_bi &goal) const override {
    return current.h_f;
  }
};

class BackwardRCPSPHeuristic : public Heuristic<RCPSPState_bi> {
public:
  double HCost(const RCPSPState_bi &goal, const RCPSPState_bi &current) const override {
    return goal.h_b;

  }
};


class RCPSP_TT : public SearchEnvironment<RCPSPState_TT,int>{
public:
  RCPSP_TT();
  void GetSuccessors(const RCPSPState_TT &nodeID, std::vector<RCPSPState_TT> &neighbors) const override;
  bool GoalTest(const RCPSPState_TT &node, const RCPSPState_TT &goal) const override;
  double HCost(const RCPSPState_TT &state1, const RCPSPState_TT &state2) const override;
  double GCost(const RCPSPState_TT &state1, const RCPSPState_TT &state2) const override;

  int GetAction(const RCPSPState_TT &nodeID, const RCPSPState_TT &nodeID2) const override;
  int GetNumSuccessors(const RCPSPState_TT &stateID) const;
  void GetActions(const RCPSPState_TT &nodeID, std::vector<int> &actions) const override;
  void ApplyAction(RCPSPState_TT &s, int a) const override;
  uint64_t GetActionHash(int act) const;
  uint64_t GetStateHash(const RCPSPState_TT &node) const;
  bool InvertAction(int &a) const;
  std::vector<RCPSPState_TT> GetSuccessors(const RCPSPState_TT &nodeID) const;
  double GCost(const RCPSPState_TT &node, const int &act) const override;
};

inline RCPSP_TT::RCPSP_TT() {


  //RCPSPex.computeAndStoreDeepDependencies();

}

inline void RCPSP_TT::GetSuccessors(const RCPSPState_TT &nodeID, std::vector<RCPSPState_TT> &neighbors) const {
  for (const auto& [transId, firingTime] : nodeID.avilableTransitionIndices) {
    neighbors.emplace_back(RCPSPState_TT(nodeID, transId, firingTime));
  }
}

inline bool RCPSP_TT::GoalTest(const RCPSPState_TT &node, const RCPSPState_TT &goal) const {
  if (node.finishedActivitiys.size() ==petri.Transitions.size())
    return true;
  else
    return false;
  // const auto& placeMarking = node.marking.at(finalstatename);
  // if (placeMarking.at("count") == 1) {
  //   return true;
  // }
  // else {
  //   return false;
  // }
}
inline double RCPSP_TT::HCost(const RCPSPState_TT &state1, const RCPSPState_TT &state2) const {
  // 9. Optimized independent set calculation
  return state1.h;


  int lastActivityId = -1;
  int maxTime = -1;

  // Find last finished activity by ID instead of name
  for (const auto& [id, time] : state1.finishedActivitiys) {
    if (time > maxTime) {
      maxTime = time;
      lastActivityId = id;
    }
  }

  if (lastActivityId != -1) {
    const std::string& lastActivityName = RCPSPex.activities[lastActivityId - 1].name;

    // Pre-reserve vectors
    std::vector<int> independentSet;
    independentSet.reserve(state1.unstartedTransitions.size());

    // Filter independent transitions
    for (int actIdx : state1.unstartedTransitions) {
      const std::string& actName = RCPSPex.activities[actIdx - 1].name;
      if (RCPSPex.deep_dependencies.find({lastActivityName, actName}) == RCPSPex.deep_dependencies.end()) {
        independentSet.push_back(actIdx);
      }
    }

    // Create lookup set for efficient filtering
    std::unordered_set<int> independentLookup(independentSet.begin(), independentSet.end());
    std::vector<int> newUnstartedTransitions;
    newUnstartedTransitions.reserve(state1.unstartedTransitions.size());

    for (int id : state1.unstartedTransitions) {
      if (independentLookup.find(id) == independentLookup.end()) {
        newUnstartedTransitions.push_back(id);
      }
    }

    // 10. Calculate heuristic efficiently
    int latestStart = 0;
    for (const auto& [id, startTime] : state1.startedActivitiys) {
      if (startTime > latestStart) {
        latestStart = startTime;
      }
    }

    int unkTime = state1.g - latestStart;
    std::map<int, int> finishedActivitiysnew=state1.finishedActivitiys;                         // activityID -> finish time
    for (int actIdx : independentSet) {
      finishedActivitiysnew[actIdx] = 0;
    }

   getForwardHcost_TT(state1.unstartedTransitions,state1.finishedActivitiys) - unkTime;
      return std::max(getForwardHcost_TT(state1.unstartedTransitions,state1.finishedActivitiys) - unkTime,
                 getForwardHcost_TT(newUnstartedTransitions,finishedActivitiysnew));
  } else {
    // Fallback if no finished activities
    return getForwardHcost_TT(state1.unstartedTransitions,state1.finishedActivitiys);
  }




}
inline double RCPSP_TT::GCost(const RCPSPState_TT &state1, const RCPSPState_TT &state2) const {
  return state2.g-state1.g;//+state1.g
}

inline uint64_t RCPSP_TT::GetStateHash(const RCPSPState_TT &node) const {
  //auto startS1 = std::chrono::high_resolution_clock::now();

  std::size_t seed = 0;

  // for (const auto& pair : node.startedActivitiys) {
  //   seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  //   seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  // }

  for (const auto& pair : node.finishedActivitiys) {
    seed ^= std::hash<int>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  //auto endS1 = std::chrono::high_resolution_clock::now();

  //hashTIME += endS1-startS1;
  return seed;

}





inline uint64_t RCPSP_TT::GetActionHash(int act) const {
  // Example hash for an action
  return std::hash<int>()(act);
}
inline void RCPSP_TT::GetActions(const RCPSPState_TT &nodeID, std::vector<int> &actions) const {
  // for (int i = 0; i < nodeID.sons.size(); ++i) {
  //   actions.push_back(i); // Add the index of each available transition as an action.
  // }
}

inline bool RCPSP_TT::InvertAction(int &a) const {
  // Example logic to invert an action
  return true;
}

inline std::vector<RCPSPState_TT> RCPSP_TT::GetSuccessors(const RCPSPState_TT &nodeID) const {

  std::vector<RCPSPState_TT> neighbors;
  // for (int i = 0; i < nodeID.sons.size(); ++i) {
  //   neighbors.push_back(nodeID.sons[i]);
  // }
  return neighbors;
}


inline int RCPSP_TT::GetAction(const RCPSPState_TT &nodeID, const RCPSPState_TT &nodeID2) const {
  return 0;
}

//inline uint64_t RCPSP::GetStateHash(const RCPSPState &s) const {
// return s.name;
//}
inline int RCPSP_TT::GetNumSuccessors(const RCPSPState_TT &stateID) const {
  // int i=0;
  // if (stateID.activeTransitions.size() > 0){i=1;}
  // return stateID.avilableTransition.size()+i;
  return 0;
}



inline void RCPSP_TT::ApplyAction(RCPSPState_TT &s, int a) const {

}
inline double RCPSP_TT::GCost(const RCPSPState_TT &node, const int &act) const {
  return node.g;
}

#endif //RCPSP_H
//
// Created by idolu on 06/01/2025.
//
