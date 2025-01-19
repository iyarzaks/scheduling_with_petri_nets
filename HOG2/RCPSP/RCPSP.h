//
// Created by idolu on 29/12/2024.
//
#ifndef RCPSP_H
#define RCPSP_H
//#include "../algorithms/OldSearchEnvironment.h"
#include "../search/SearchEnvironment.h"
#include "RCPSPState.h"
//creted the RCPSPState in searchgraph
// class RCPSPState{
// searchNode node;
//   };
std::uint64_t count=0;
//int Nsize=0;
class action {
public:
int a=0;
};
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
inline double RCPSP::GCost(const RCPSPState &node, const int &act) const {
  return node.g;
}

inline uint64_t RCPSP::GetStateHash(const RCPSPState &node) const {
  // Example hash: combine state name and some state property
  return std::hash<int>()(node.name) ^ std::hash<int>()(node.g);
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
    // Transition active = nodeID.activeTransitions[t];
    // RCPSPState temp(RCPSPState(nodeID,active,0,t,count));
    // temp.name = count;
    // neighbors.emplace_back(temp);
    //Nsize++;

    neighbors.emplace_back(RCPSPState(nodeID,nodeID.activeTransitions[t],0,t,count));
  }

  for (int i=0;i<nodeID.avilableTransition.size();i++) {
    count++;
    //RCPSPState temp(nodeID,nodeID.avilableTransition[i],1,i,count);
    //temp.name = count;
    neighbors.emplace_back(RCPSPState(nodeID,nodeID.avilableTransition[i],1,i,count));
    //Nsize++;
  }
//   for (int i=0; i<nodeID.sons.size(); i++) {
//   neighbors.emplace_back(nodeID.sons[i]);
// }
}

inline bool RCPSP::GoalTest(const RCPSPState &node, const RCPSPState &goal) const {
  if (node.marking.at(finalstatename) == goal.marking.at(finalstatename)) {
  return true;
}
  return false;
}

inline double RCPSP::HCost(const RCPSPState &state1, const RCPSPState &state2) const {
//return state1.h-state2.h;
return state2.h;
}

inline double RCPSP::GCost(const RCPSPState &state1, const RCPSPState &state2) const {
  //return state1.g-state2.g;
  return state2.g;
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
  a = -a; // Negate the action (depends on your domain).
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
  return 0; // Placeholder. Adjust this logic to your needs.
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

  // if (s.activeTransitions.size()>0) {
  //   count++;
  //   int t=0;
  //   for (int i=0;i<s.activeTransitions.size();i++) {
  //     if (s.activeTransitions[i].duration<s.activeTransitions[t].duration) {
  //       t=i;
  //     }
  //   }
  //   Transition active = s.activeTransitions[t];
  //   s.sons.push_back(RCPSPState(s,active,0,t,count));
  // }
  //
  // for (int i=0;i<s.avilableTransition.size();i++) {
  //   count++;
  //   s.sons.push_back(RCPSPState(s,s.avilableTransition[i],1,i,count));
  //   s.sons.back().name = count;
  //
  // }
}


#endif //RCPSP_H
//
// Created by idolu on 06/01/2025.
//
