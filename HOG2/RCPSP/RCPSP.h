//
// Created by idolu on 29/12/2024.
//

#ifndef RCPSP_H
#define RCPSP_H

#include "SearchEnvironment.h"
#include "searchgraph.h"
//creted the RCPSPState in searchgraph
// class RCPSPState{
// searchNode node;
//   };
int count=0;
class action {
public:
int a=0;
};
class RCPSP : public SearchEnvironment<RCPSPState,int>{
  public:
  void GetSuccessors(const RCPSPState &nodeID, std::vector<RCPSPState> &neighbors) const override;
  bool GoalTest(const RCPSPState &node) const override;
	double HCost(const RCPSPState &state1, const RCPSPState &state2) const override;
	double GCost(const RCPSPState &state1, const RCPSPState &state2) const override;
  int GetAction(const RCPSPState &nodeID, const RCPSPState &nodeID2) const override;
  int GetNumSuccessors(const RCPSPState &stateID) const;
  void GetActions(const RCPSPState &nodeID, std::vector<int> &actions) const override;
  void ApplyAction(RCPSPState &s, int a) const override;

  };



inline void RCPSP::GetSuccessors(const RCPSPState &nodeID, std::vector<RCPSPState> &neighbors) const {
for (int i=0; i<nodeID.sons.size(); i++) {
  neighbors.push_back(nodeID.sons[i]);
}
}

inline bool RCPSP::GoalTest(const RCPSPState &node) const {
  if (node.marking.at(node.finalstatename) == 1) {
  return true;
}
  return false;
}

inline double RCPSP::HCost(const RCPSPState &state1, const RCPSPState &state2) const {
return 0;
}

inline double RCPSP::GCost(const RCPSPState &state1, const RCPSPState &state2) const {
  return state1.g;
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

inline void RCPSP::GetActions(const RCPSPState &nodeID, std::vector<int> &actions) const {
}


inline void RCPSP::ApplyAction(RCPSPState &s, int a) const {
  if (s.activeTransitions.size()>0) {
    count++;
    int t=0;
    for (int i=0;i<s.activeTransitions.size();i++) {
      if (s.activeTransitions[i].duration<s.activeTransitions[t].duration) {
        t=i;
      }
    }
    Transition active = s.activeTransitions[t];
    s.sons.push_back(RCPSPState(s,active,0,t,count));
  }

  for (int i=0;i<s.avilableTransition.size();i++) {
    count++;
    s.sons.push_back(RCPSPState(s,s.avilableTransition[i],1,i,count));
    s.sons.back().name = count;

  }
}

#endif //RCPSP_H
