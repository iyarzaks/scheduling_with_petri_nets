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


inline void hash_combine(size_t& seed, size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash function for std::map<std::string, int>
size_t hash_map(const std::map<std::string, int>& m) {
  size_t seed = 0;
  for (const auto& [key, value] : m) {
    hash_combine(seed, std::hash<std::string>()(key));
    hash_combine(seed, std::hash<int>()(value));
  }
  return seed;
}
struct TransitionHash {
  size_t operator()(const Transition& t) const {
    size_t seed = 0;
    hash_combine(seed, std::hash<std::string>()(t.name));
    hash_combine(seed, std::hash<int>()(t.duration));
    hash_combine(seed, hash_map(t.arcs_in));
    hash_combine(seed, hash_map(t.arcs_out));
    return seed;
  }
};
inline uint64_t RCPSP::GetStateHash(const RCPSPState &node) const {

  constexpr uint64_t PRIME = 0x100000001b3;
  uint64_t hash = 0xcbf29ce484222325;

//  Hash g and h values
  // hash ^= static_cast<uint64_t>(node.g);
  // hash *= PRIME;
  // hash ^= static_cast<uint64_t>(node.h);
  // hash *= PRIME;

  // For marking values, since they're mostly binary,
  // we can pack multiple values into one hash operation
  uint64_t markingBits = 0;
  int bitPos = 0;
  for (const auto& pair : node.marking) {
    if (pair.second <= 1) {
      // For binary values, use bit packing
      if (pair.second == 1) {
        markingBits |= (1ULL << bitPos);
      }
      bitPos++;
      if (bitPos == 64) {
        // If we fill up 64 bits, hash them and reset
        hash ^= markingBits;
        hash *= PRIME;
        markingBits = 0;
        bitPos = 0;
      }
    } else {
      // For non-binary values, hash them directly
      hash ^= static_cast<uint64_t>(pair.second);
      hash *= PRIME;
    }
  }
  // Hash any remaining marking bits
  if (bitPos > 0) {
    hash ^= markingBits;
    hash *= PRIME;
  }

  // Hash active transitions in order-independent way
  uint64_t transitionsHash = 0;
  for (const Transition& trans : node.activeTransitions) {
    uint64_t transHash = std::hash<std::string>{}(trans.name);
    transHash ^= static_cast<uint64_t>(trans.duration);
    transHash *= PRIME;
    transitionsHash ^= transHash;
  }
  hash ^= transitionsHash;
  hash *= PRIME;

  return hash;


  // Combine multiple state properties for a more robust hash
  /*size_t hash = 0;

  // Hash integers
  hash ^= std::hash<int>()(node.g) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= std::hash<int>()(node.h) + 0x9e3779b9 + (hash << 6) + (hash >> 2);

  // Hash marking (std::map<std::string, int>)
  for (const auto& [key, value] : node.marking) {
    hash ^= std::hash<std::string>()(key) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>()(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }

  // Hash unstartedTransitions (std::vector<int>)
  for (int val : node.unstartedTransitions) {
    hash ^= std::hash<int>()(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }

  // Hash activeTransitions (std::vector<Transition>)
  for (const auto& t : node.activeTransitions) {
    hash ^= TransitionHash()(t) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }

  // Hash availableTransitions (std::vector<Transition>)
  for (const auto& t : node.avilableTransition) {
    hash ^= TransitionHash()(t) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }

  return hash;


  */

}


inline RCPSP::RCPSP() {
}

inline void RCPSP::GetSuccessors(const RCPSPState &nodeID, std::vector<RCPSPState> &neighbors) const {
static int exnum;
  exnum++;
  if (exnum == 1000000) {exit;}
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
inline double RCPSP::GCost(const RCPSPState &node, const int &act) const {
  return node.g;
}

#endif //RCPSP_H
//
// Created by idolu on 06/01/2025.
//
