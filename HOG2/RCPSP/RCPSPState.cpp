//
// Created by idolu on 06/01/2025.

#include <iostream>
#include <vector>
#include <set>
#include "RCPSPState.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
std::chrono::duration<double> generateTIME;
std::chrono::duration<double> avelableTIME;
std::chrono::duration<double> HTIME;

// std::atomic<bool> stop_printing(false); // Flag to stop the printing thread
//
// void printNetworkSize(const std::vector<RCPSPState>& network) {
//   while (!stop_printing) {
//     std::this_thread::sleep_for(std::chrono::seconds(1)); // Wait for a second
//     std::cout << "Current network size: " << network.size() << std::endl;
//   }
// }
//std::vector<Transition> getAvilableTransitions(std::map<std::string, int> marking);

std::vector<Transition> getAvilableTransitions(const std::unordered_map<std::string, int>& marking);

void GetNabor(std::vector<RCPSPState> &NodeList,int chosenNode,int &count);
//int ChooseExpansion(std::vector<RCPSPState> network);
 PetriExample petri;
 RCPSP_example RCPSPex;
 int main2() {
   return 0;
 }
std::vector<Transition> getAvilableDetransitions(const std::unordered_map<std::string, int>& marking) {
   std::vector<Transition> availableDetransitions;
   availableDetransitions.reserve(petri.Transitions.size());

   for (const auto& transition : petri.Transitions) {
     bool canUndo = true;

     for (const auto& arc : transition.arcs_out) {  // Instead of arcs_in, we check arcs_out
       auto it = marking.find(arc.first);
       int tokenCount = (it != marking.end()) ? it->second : 0;

       if (tokenCount < arc.second) {
         canUndo = false;  // Not enough tokens in the output place to undo
         break;
       }
     }

     if (canUndo) {
       availableDetransitions.push_back(transition);
     }
   }

   return availableDetransitions;


 }
std::vector<Transition> getAvilableTransitions(const std::unordered_map<std::string, int>& marking) {
   auto startS1 = std::chrono::high_resolution_clock::now();

   std::vector<Transition> avilableTransitions;
   avilableTransitions.reserve(petri.Transitions.size());  // Reserve memory to avoid multiple reallocations

   for (const auto& transition : petri.Transitions) {
     int avilable = 0, requirment = 0;
     bool canFire = true;

     for (const auto& arc : transition.arcs_in) {
       auto it = marking.find(arc.first);
       int tokenCount = (it != marking.end()) ? it->second : 0;

       if (tokenCount < arc.second) {
         canFire = false;  // Not enough tokens to fire
         break;            // Stop checking further
       }
       avilable += std::min(tokenCount, arc.second);
       requirment += arc.second;
     }

     if (canFire) {
       avilableTransitions.push_back(transition);
     }
   }

   auto endS1 = std::chrono::high_resolution_clock::now();
   avelableTIME += endS1 - startS1;

   return avilableTransitions;
 }

std::vector<int> getAvilableTransitionIndices(const std::unordered_map<std::string, int>& marking);
std::vector<int> getAvilableDetransitionIndices(const std::unordered_map<std::string, int>& marking);



/*
void GetNabor(std::vector<RCPSPState> &NodeList,int chosenNode,uint64_t &count) {
  if (NodeList[chosenNode].activeTransitions.size()>0) {
    count++;
    int t=0;
    for (int i=0;i<NodeList[chosenNode].activeTransitions.size();i++) {
      if (NodeList[chosenNode].activeTransitions[i].duration<NodeList[chosenNode].activeTransitions[t].duration) {
        t=i;
      }
    }
    Transition active = NodeList[chosenNode].activeTransitions[t];
    NodeList.push_back(RCPSPState(NodeList[chosenNode],active,0,t,count));
  }

  for (int i=0;i<NodeList[chosenNode].avilableTransition.size();i++) {
    count++;
    NodeList.push_back(RCPSPState(NodeList[chosenNode],NodeList[chosenNode].avilableTransition[i],1,i,count));
    //NodeList.back().name = count;

 }
}
*/
double getForwardHcost(std::vector<int>unstartedTransitions, std::vector<std::pair<int, int>>activeTransitionIndices) {
  auto startS3 = std::chrono::high_resolution_clock::now();

   std::map<int, int> earlyfinishMap2; // Map to store activity IDs and their early finish times
  //std::map<int, int> visitmap; // Map to store activity IDs and their early finish times
  double h;
  std::set<int> processedDependencies;
  // Iterate over unstarted activitiesint lastElementEarlyFinish = 0;
  //int lastElementEarlyFinish = 0;
  for (int activityId: unstartedTransitions) {
    int maxFinishTime = 0;
    std::set<int> processedDependencies;

    for (const auto &dep: RCPSPex.backword_dependencies[activityId - 1]) {
      int depId = std::stoi(dep) - 1;
      // if (processedDependencies.count(depId) > 0) continue;
      // processedDependencies.insert(depId);
      if (std::find(unstartedTransitions.begin(), unstartedTransitions.end(), depId + 1) != unstartedTransitions.end()) {
        int duration = getTransitionDuration2(activeTransitionIndices, std::stoi(dep));
        if (duration !=-1) {
          maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1] + duration);
          //if (RCPSPex.activities[depId].duration !=duration) {
          //  std::cout<<name<<":"<<dep<<" "<<activityId<<" "<<RCPSPex.activities[depId].duration-duration<<std::endl;
          //}
        }
        else {
          maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1] + RCPSPex.activities[depId].duration);

        }
      }
      else {
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1]);
      }
    }

    earlyfinishMap2[activityId] = maxFinishTime;
    //std::cout <<activityId<<":"<< earlyfinishMap[activityId]+RCPSPex.activities[activityId-1].duration << std::endl;
    // For last element with duration 0, just use the max finish time of dependencies
  }
  if (earlyfinishMap2.size()==0) {
    h = 0;
  }
  else {
    h = earlyfinishMap2.rbegin()->second;;

  }
  // if (h != newH) {
  //   int asd;
  //   asd++;
  // }
   auto endS3 = std::chrono::high_resolution_clock::now();
   HTIME += endS3 - startS3;

 return h;

}
double getBackwordsHcost(std::vector<int> completedActivities,
                           std::vector<std::pair<int, int>> activeTransitionIndices,
                           int active) {
  auto startS3 = std::chrono::high_resolution_clock::now();
return 0;
  std::map<int, int> earlyStartMap; // Map to store activity IDs and their early start times
  std::map<int, int> earlyFinishMap; // Map to store activity IDs and their early finish times

  // Initialize all activities with early start = 0
  for (int i = 0; i < RCPSPex.activities.size(); i++) {
    earlyStartMap[i + 1] = 0;
    earlyFinishMap[i + 1] = 0;
  }

  // Process activities in topological order
  std::vector<int> toProcess;
  std::set<int> processed;

  // First, add activities with no dependencies (project start activities)
  for (int i = 0; i < RCPSPex.activities.size(); i++) {
    if (RCPSPex.dependencies[i].empty()) {
      toProcess.push_back(i + 1);
    }
  }

  // Process activities in topological order
  while (!toProcess.empty()) {
    int currentActivity = toProcess.front();
    toProcess.erase(toProcess.begin());

    if (processed.count(currentActivity) > 0) continue;

    // Check if all predecessors have been processed
    bool allPredecessorsProcessed = true;
    for (const auto& pred : RCPSPex.backword_dependencies[currentActivity - 1]) {
      int predId = std::stoi(pred);
      if (processed.count(predId) == 0) {
        allPredecessorsProcessed = false;
        break;
      }
    }

    if (!allPredecessorsProcessed) {
      toProcess.push_back(currentActivity);
      continue;
    }

    // Calculate early start time (maximum of all predecessors' early finish times)
    int maxPredFinish = 0;
    for (const auto& pred : RCPSPex.backword_dependencies[currentActivity - 1]) {
      int predId = std::stoi(pred);
      maxPredFinish = std::max(maxPredFinish, earlyFinishMap[predId]);
    }

    earlyStartMap[currentActivity] = maxPredFinish;

    // Calculate early finish time
    int duration;
    // Check if this activity has a modified duration in activeTransitionIndices
    auto it = std::find_if(activeTransitionIndices.begin(), activeTransitionIndices.end(),
                        [currentActivity](const std::pair<int, int>& p) { return p.first == currentActivity; });

    if (it != activeTransitionIndices.end()) {
      // Use the remaining duration from activeTransitionIndices
      duration = it->second;
    } else {
      // Use the full duration from activities list
      duration = RCPSPex.activities[currentActivity - 1].duration;
    }

    earlyFinishMap[currentActivity] = earlyStartMap[currentActivity] + duration;

    // Add successors to the processing queue
    for (const auto& succ : RCPSPex.dependencies[currentActivity - 1]) {
      int succId = std::stoi(succ);
      toProcess.push_back(succId);
    }

    processed.insert(currentActivity);

    // If we've processed the active activity, we can stop
    if (currentActivity == active) {
      break;
    }
  }

  auto endS3 = std::chrono::high_resolution_clock::now();
  HTIME += endS3 - startS3;

  // Return early finish time of active activity
  return earlyFinishMap[active];
}
RCPSPState::RCPSPState(): nodestatus(false) {
  auto startS1 = std::chrono::high_resolution_clock::now();

  direction = true;
  startedActivitiys[0] = 0;

  // Find initial and final places
  for (int i = 0; i < petri.places.size(); i++) {
    if (petri.places[i].arcs_out.size() == 0) {
      finalstatename = petri.places[i].name;
    }
    if (petri.places[i].arcs_in.size() == 0) {
      initialstatename = petri.places[i].name;
    }
  }

  // Initialize unstartedTransitions
  for (int i = 1; i < petri.Transitions.size(); i++) {
    unstartedTransitions.push_back(i + 1);
  }

  // Initialize marking
  for (int i = 0; i < petri.places.size(); i++) {
    if (petri.places[i].name == initialstatename) {
      marking[petri.places[i].name] = 1;
    } else {
      marking[petri.places[i].name] = petri.places[i].state[0][0];
    }
  }

  auto endS1 = std::chrono::high_resolution_clock::now();
  generateTIME += endS1 - startS1;

  // Change: Get indices of available transitions instead of full Transition objects
  avilableTransitionIndices = getAvilableTransitionIndices(marking);

  g = 0;
  name = 0;
}

RCPSPState_bi::RCPSPState_bi(): nodestatus(false) {
  auto startS1 = std::chrono::high_resolution_clock::now();

  direction = true;
  startedActivitiys[0] = 0;

  // Find initial and final places
  for (int i = 0; i < petri.places.size(); i++) {
    if (petri.places[i].arcs_out.size() == 0) {
      finalstatename = petri.places[i].name;
    }
    if (petri.places[i].arcs_in.size() == 0) {
      initialstatename = petri.places[i].name;
    }
  }

  // Initialize unstartedTransitions
  for (int i = 1; i < petri.Transitions.size(); i++) {
    unstartedTransitions.push_back(i + 1);
  }

  // Initialize marking
  for (int i = 0; i < petri.places.size(); i++) {
    if (petri.places[i].name == initialstatename) {
      marking[petri.places[i].name] = 1;
    } else {
      marking[petri.places[i].name] = petri.places[i].state[0][0];
    }
  }

  auto endS1 = std::chrono::high_resolution_clock::now();
  generateTIME += endS1 - startS1;

  // Change: Get indices of available transitions instead of full Transition objects
  avilableTransitionIndices = getAvilableTransitionIndices(marking);

  avilableDeTransitionIndices = getAvilableDetransitionIndices(marking);
  g = 0;
  name = 0;
  h=getForwardHcost(unstartedTransitions,activeTransitionIndices);
}


RCPSPState::RCPSPState(RCPSPState predecesor, Transition active, bool status, int location, uint64_t &count) {
  auto startS4 = std::chrono::high_resolution_clock::now();

  // Copy basic properties
  direction = predecesor.direction;
  name = count;
  nodestatus = status;
  unstartedTransitions = predecesor.unstartedTransitions;
  startedActivitiys = predecesor.startedActivitiys;
  finishedActivitiys = predecesor.finishedActivitiys;
  marking = predecesor.marking;

  // Copy indices instead of full Transition objects
  activeTransitionIndices = predecesor.activeTransitionIndices;
  avilableTransitionIndices = predecesor.avilableTransitionIndices;
  g = predecesor.g;



  if (direction) {
    if (status) {
      h = predecesor.h;

      // Apply arcs_in from the transition
      for (const auto& arc : petri.Transitions[active.name-1].arcs_in) {
        marking[arc.first] -= arc.second;
      }

      // Store index and duration instead of full Transition
      activeTransitionIndices.push_back({active.name, active.duration});
      startedActivitiys[active.name] = g;
      if (active.duration==0) {
        status=0;
      }
      auto endS1 = std::chrono::high_resolution_clock::now();
      generateTIME += endS1-startS4;
    }
    if (!status) {
      g += active.duration;
      finishedActivitiys[active.name] = g;

      // Remove from unstarted
      unstartedTransitions.erase(
          std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), active.name),
          unstartedTransitions.end());

      // Update durations and remove completed transitions
      for (int i = activeTransitionIndices.size() - 1; i >= 0; --i) {
        activeTransitionIndices[i].second -= active.duration;
        if (activeTransitionIndices[i].second <0) {
          activeTransitionIndices[i].second =0;

        }
        if (activeTransitionIndices[i].first == active.name) {
          // Apply arcs_out from the transition
          for (const auto& arc : petri.Transitions[active.name-1].arcs_out) {
            marking[arc.first] += arc.second;
          }
          activeTransitionIndices.erase(activeTransitionIndices.begin() + i);
        }

      }

      auto endS1 = std::chrono::high_resolution_clock::now();
      generateTIME += endS1-startS4;

      h=getForwardHcost(unstartedTransitions,activeTransitionIndices);

    }
  }
  else {
    // Similar transformation for the backward direction
    if (status) {
      h = predecesor.h;

      for (const auto& arc : petri.Transitions[active.name-1].arcs_out) {
        marking[arc.first] -= arc.second;
      }

      activeTransitionIndices.push_back({active.name, active.duration});
      startedActivitiys[active.name] = g;

      auto endS1 = std::chrono::high_resolution_clock::now();
      generateTIME += endS1-startS4;
    }
    else {
      g += active.duration;
      finishedActivitiys[active.name] = g;

      unstartedTransitions.erase(
          std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), active.name),
          unstartedTransitions.end());

      for (int i = activeTransitionIndices.size() - 1; i >= 0; --i) {
        activeTransitionIndices[i].second -= active.duration;
        if (activeTransitionIndices[i].second<0) {
          activeTransitionIndices[i].second=0;
        }
        if (activeTransitionIndices[i].first == active.name) {
          for (const auto& arc : petri.Transitions[active.name-1].arcs_in) {
            marking[arc.first] += arc.second;
          }
          activeTransitionIndices.erase(activeTransitionIndices.begin() + i);
        }
      }
      auto endS1 = std::chrono::high_resolution_clock::now();
      generateTIME += endS1-startS4;
      h=getBackwordsHcost(unstartedTransitions,activeTransitionIndices,active.name);

    }
  }
  avilableTransitionIndices = getAvilableTransitionIndices(marking);



  // You'll need to modify these functions to return indices instead of Transitions
//   if (direction) {
// }
// else {
//   }
int asdasd;
  asdasd++;
}


RCPSPState_bi::RCPSPState_bi(RCPSPState_bi predecesor, Transition active, bool status, int location, uint64_t &count) {
  auto startS4 = std::chrono::high_resolution_clock::now();

  // Copy basic properties
  direction = predecesor.direction;
  name = count;
  nodestatus = status;
  unstartedTransitions = predecesor.unstartedTransitions;
  startedActivitiys = predecesor.startedActivitiys;
  finishedActivitiys = predecesor.finishedActivitiys;
  marking = predecesor.marking;

  // Copy indices instead of full Transition objects
  activeTransitionIndices = predecesor.activeTransitionIndices;
  avilableTransitionIndices = predecesor.avilableTransitionIndices;
  g = predecesor.g;



  if (direction) {
    if (status) {
      h = predecesor.h;

      // Apply arcs_in from the transition
      for (const auto& arc : petri.Transitions[active.name-1].arcs_in) {
        marking[arc.first] -= arc.second;
      }

      // Store index and duration instead of full Transition
      activeTransitionIndices.push_back({active.name, active.duration});
      startedActivitiys[active.name] = g;
      auto endS1 = std::chrono::high_resolution_clock::now();
      generateTIME += endS1-startS4;
    }
    else {
      g += active.duration;
      finishedActivitiys[active.name] = g;

      // Remove from unstarted
      unstartedTransitions.erase(
          std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), active.name),
          unstartedTransitions.end());

      // Update durations and remove completed transitions
      for (int i = activeTransitionIndices.size() - 1; i >= 0; --i) {
        activeTransitionIndices[i].second -= active.duration;
        if (activeTransitionIndices[i].second <0) {
          activeTransitionIndices[i].second =0;

        }
        if (activeTransitionIndices[i].first == active.name) {
          // Apply arcs_out from the transition
          for (const auto& arc : petri.Transitions[active.name-1].arcs_out) {
            marking[arc.first] += arc.second;
          }
          activeTransitionIndices.erase(activeTransitionIndices.begin() + i);
        }

      }

      auto endS1 = std::chrono::high_resolution_clock::now();
      generateTIME += endS1-startS4;

      h=getForwardHcost(unstartedTransitions,activeTransitionIndices);

    }
  }
  else {
    // Similar transformation for the backward direction
    if (status) {
      h = predecesor.h;

      for (const auto& arc : petri.Transitions[active.name-1].arcs_out) {
        marking[arc.first] -= arc.second;
      }

      activeTransitionIndices.push_back({active.name, active.duration});
      finishedActivitiys[active.name] = g;

      auto endS1 = std::chrono::high_resolution_clock::now();
      generateTIME += endS1-startS4;
    }
    else {
      g += active.duration;
      startedActivitiys[active.name] = g;

      unstartedTransitions.erase(
          std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), active.name),
          unstartedTransitions.end());

      for (int i = activeTransitionIndices.size() - 1; i >= 0; --i) {
        activeTransitionIndices[i].second -= active.duration;
        if (activeTransitionIndices[i].second<0) {
          activeTransitionIndices[i].second=0;
        }
        if (activeTransitionIndices[i].first == active.name) {
          for (const auto& arc : petri.Transitions[active.name-1].arcs_in) {
            marking[arc.first] += arc.second;
          }
          activeTransitionIndices.erase(activeTransitionIndices.begin() + i);
        }
      }
      auto endS1 = std::chrono::high_resolution_clock::now();
      generateTIME += endS1-startS4;
      h=getForwardHcost(unstartedTransitions,activeTransitionIndices);

    }
  }
  avilableTransitionIndices = getAvilableTransitionIndices(marking);



  avilableDeTransitionIndices = getAvilableDetransitionIndices(marking);
  // You'll need to modify these functions to return indices instead of Transitions
//   if (direction) {
// }
// else {
//   }
int asdasd;
  asdasd++;
}
std::vector<int> getAvilableTransitionIndices(const std::unordered_map<std::string, int>& marking) {
  auto startS4 = std::chrono::high_resolution_clock::now();

  std::vector<int> availableIndices;

   // Loop through all transitions (assuming they're indexed starting from 1)
   for (int i = 0; i < petri.Transitions.size(); i++) {
     const Transition& t = petri.Transitions[i];
     bool available = true;

     // Check if all input arcs have sufficient tokens
     for (const auto& arc : t.arcs_in) {
       auto it = marking.find(arc.first);
       if (it == marking.end() || it->second < arc.second) {
         available = false;
         break;
       }
     }

     if (available) {
       availableIndices.push_back(i + 1);  // +1 assuming your indices start from 1
     }
   }
  auto endS1 = std::chrono::high_resolution_clock::now();
  avelableTIME += endS1-startS4;
   return availableIndices;
 }

std::vector<int> getAvilableDetransitionIndices(const std::unordered_map<std::string, int>& marking) {
   std::vector<int> availableIndices;

   // Similar implementation for detransitions
   for (int i = 0; i < petri.Transitions.size(); i++) {
     const Transition& t = petri.Transitions[i];
     bool available = true;

     // Check output arcs instead of input arcs for detransitions
     for (const auto& arc : t.arcs_out) {
       auto it = marking.find(arc.first);
       if (it == marking.end() || it->second < arc.second) {
         available = false;
         break;
       }
     }

     if (available) {
       availableIndices.push_back(i + 1);
     }
   }

   return availableIndices;
 }



bool RCPSPState::operator==(const RCPSPState &other) const {
   // if (this->expanded != other.expanded) {
   //   return false;
   // }
   if (this->g != other.g) {
     return false;
   }
   if (this->h != other.h) {
     return false;
   }
   // if (this->avilableTransition != other.avilableTransition) {
   //   return false;
   // }
   // if (this->activeTransitions != other.activeTransitions) {
   //   return false;
   // }
   if (this->unstartedTransitions != other.unstartedTransitions) {
     return false;
   }
   if (this->marking != other.marking) {
     return false;
   }
   return true;
 }
bool RCPSPState_bi::operator==(const RCPSPState_bi &other) const {
  // if (this->expanded != other.expanded) {
  //   return false;
  // }
  if (this->g != other.g) {
    return false;
  }
  if (this->h != other.h) {
    return false;
  }
  // if (this->avilableTransition != other.avilableTransition) {
  //   return false;
  // }
  // if (this->activeTransitions != other.activeTransitions) {
  //   return false;
  // }
  if (this->unstartedTransitions != other.unstartedTransitions) {
    return false;
  }
  if (this->marking != other.marking) {
    return false;
  }
  return true;
}