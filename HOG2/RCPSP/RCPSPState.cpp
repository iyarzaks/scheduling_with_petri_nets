//
// Created by idolu on 06/01/2025.

#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>

#include "RCPSPState.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
//std::chrono::duration<double> generateTIME;
//std::chrono::duration<double> avelableTIME;
//std::chrono::duration<double> HTIME;

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
double computeWorkloadLowerBoundWithMax(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    const std::map<int, int>& earlyStartTimes,
    double criticalPathEstimate
);
std::vector<std::pair<int, int>> getAvailableTransitionIndices_TT(
    const std::vector<int> &unstartedTransitions,
    const std::map<int, int> &finishedActivities,
    const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &marking
);
std::vector<int> getCriticalPath(const std::map<int, int>& earlyStartTimes,
                                 int lastActivityId,
                                 const std::vector<std::vector<std::string>>& backword_dependencies);


double computeSequenceLowerBoundWithMax(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    const std::map<int, int>& earlyStartTimes,
    double criticalPathEstimate
);
double computeSequenceLowerBoundWithMax2(
  const std::vector<int>& unfinishedTransitions,
const std::vector<std::pair<int, int>>& activeTransitionIndices,
 std::map<int, int>& earlyStartTimes,
 std::map<int, int>& earlyfinishTimes,
double criticalPathEstimate,
std::map<int, int> finishedActivities
);
double computeCoreTimeLowerBoundWithMax(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    const std::map<int, int>& earlyStartTimes,
    double criticalPathEstimate
);
double computeResourceCapacityLowerBound(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    double criticalPathEstimate
);

double getBackwardHcost2(
    const std::set<int>& startedActivities,
    const std::set<int>& finishedActivities,
    const std::vector<std::pair<int, int>>& activeTransitionIndices
);


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
   //auto startS1 = std::chrono::high_resolution_clock::now();

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

   // auto endS1 = std::chrono::high_resolution_clock::now();
   // avelableTIME += endS1 - startS1;

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
double getForwardHcost(std::set<int>unstartedTransitions, std::vector<std::pair<int, int>>activeTransitionIndices) {
 // auto startS3 = std::chrono::high_resolution_clock::now();

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
  // auto endS3 = std::chrono::high_resolution_clock::now();
   //HTIME += endS3 - startS3;

 return h;

}



double getForwardHcost(std::vector<int>unstartedTransitions,
                      std::vector<std::pair<int, int>>activeTransitionIndices,
                      std::map<int, int> finishedActivities = {{0, 0}}
                      ) {
  //auto startS3 = std::chrono::high_resolution_clock::now();

   std::map<int, int> earlyfinishMap2; // Map to store activity IDs and their early finish times
   std::map<int, int> earlyfinishMap3; // Map to store activity IDs and their early finish times
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
          earlyfinishMap3[depId+1] = earlyfinishMap2[depId+1] + duration;
          //if (RCPSPex.activities[depId].duration !=duration) {
          //  std::cout<<name<<":"<<dep<<" "<<activityId<<" "<<RCPSPex.activities[depId].duration-duration<<std::endl;
          //}
        }
        else {
          maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1] + RCPSPex.activities[depId].duration);
          earlyfinishMap3[depId+1] = earlyfinishMap2[depId+1] + RCPSPex.activities[depId].duration;
        }
      }
      else {
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1]);
        earlyfinishMap3[depId+1] = earlyfinishMap2[depId+1];
      }
    }

    earlyfinishMap2[activityId] = maxFinishTime;
    earlyfinishMap3[activityId] = maxFinishTime;
    //std::cout <<activityId<<":"<< earlyfinishMap[activityId]+RCPSPex.activities[activityId-1].duration << std::endl;
    // For last element with duration 0, just use the max finish time of dependencies
  }
  if (earlyfinishMap2.size()==0) {
    h = 0;
  }
  else {
    h = earlyfinishMap2.rbegin()->second;;

  }
  if (1) {
    return computeSequenceLowerBoundWithMax2(
        unstartedTransitions,
        activeTransitionIndices,
        earlyfinishMap2,
        earlyfinishMap3,
        h,
        finishedActivities); // BL_Cs heuristic
  } else {
    return h;
  }
//return h;
 // return std::max(computeResourceCapacityLowerBound(unstartedTransitions,activeTransitionIndices,h), computeSequenceLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h));//BL_RC huristic
  //return computeResourceCapacityLowerBound(unstartedTransitions,activeTransitionIndices,h);//BL_Cs huristic
  //return computeSequenceLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_Cs huristic
  //return computeCoreTimeLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_CT huristic
  //return computeWorkloadLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_CC huristic
///
 return h;

}
std::vector<int> getCriticalPath(const std::map<int, int>& earlyfinishTimes,
                                 int lastActivityId) {
    std::vector<int> criticalPath;

    // Input validation
    if (earlyfinishTimes.empty() || lastActivityId <= 0) {
        return criticalPath;
    }

    // Check if lastActivityId exists in earlyStartTimes
    if (earlyfinishTimes.find(lastActivityId) == earlyfinishTimes.end()) {
        return criticalPath;
    }

    std::set<int> visited; // Prevent infinite loops
    int current = lastActivityId;

    while (current != -1) {
        // Check for cycles
        if (visited.count(current)) {
            break; // Cycle detected, stop to prevent infinite loop
        }

        visited.insert(current);
        criticalPath.push_back(current);

        // Get dependencies for current activity
        // Check bounds for RCPSPex.backword_dependencies array
        if (current < 1 || current > static_cast<int>(RCPSPex.backword_dependencies.size())) {
            break;
        }

        const auto& deps = RCPSPex.backword_dependencies[current - 1]; // Convert to 0-based

        if (deps.empty()) {
            break; // No more predecessors
        }

        int maxearlyfinish = -1;
        int nextActivity = -1;

        // Find predecessor with maximum early start time
        for (const std::string& depStr : deps) {
            // Validate string conversion
            if (depStr.empty()) continue;

            int depId = -1;
            try {
                depId = std::stoi(depStr);
            } catch (const std::exception&) {
                continue; // Skip invalid string
            }

            // Validate depId
            if (depId <= 0) continue;

            // Check if this dependency exists in earlyStartTimes
            auto it = earlyfinishTimes.find(depId);
            if (it == earlyfinishTimes.end()) continue;

            int earlyfinish = it->second;

            // Select predecessor with maximum early start time
            if (earlyfinish > maxearlyfinish) {
                maxearlyfinish = earlyfinish;
                nextActivity = depId;
            }
           if (earlyfinish==0) {
          //   visited.insert(current);
          //   criticalPath.push_back(current);
            break;
          }
        }
        // Update current for next iteration
        current = nextActivity;
    }

    // Reverse to get path from start to end
    std::reverse(criticalPath.begin(), criticalPath.end());

    return criticalPath;
}
double getBackwardHcost2(
    const std::set<int>& startedActivities,
    const std::set<int>& finishedActivities,
    const std::vector<std::pair<int, int>>& activeTransitionIndices
) {
  std::map<int, int> earlyFinishMap;
  std::set<int> allRelevant;

  for (int id : startedActivities)
    allRelevant.insert(id);
  for (const auto& [id, _] : activeTransitionIndices)
    allRelevant.insert(id);

  for (int actId : allRelevant) {
    int maxDepFinish = 0;
    for (const std::string& depStr : RCPSPex.backword_dependencies[actId - 1]) {
      int depId = std::stoi(depStr);
      if (earlyFinishMap.count(depId))
        maxDepFinish = std::max(maxDepFinish, earlyFinishMap[depId]);
    }

    int duration = RCPSPex.activities[actId - 1].duration;
    int remaining = 0;
    for (const auto& [id, remain] : activeTransitionIndices) {
      if (id == actId) {
        remaining = remain;
        break;
      }
    }

    int effectiveDuration = duration - remaining;
    earlyFinishMap[actId] = maxDepFinish + effectiveDuration;
  }

  int maxSoFar = 0;
  for (const auto& [_, finishTime] : earlyFinishMap)
    maxSoFar = std::max(maxSoFar, finishTime);

  return static_cast<double>(maxSoFar);
}

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
std::map<int, int> getlatefinish (int size,int finishTime) {
std::map<int, int> latefinishTimes;

    // Set the last activity's late finish time
    latefinishTimes[size] = finishTime;

    // Work backwards from the last activity
    for (int i = size - 1; i > 0; i--) {
        // Check bounds for forward_dependencies array
        if (i < 1 || i > static_cast<int>(RCPSPex.dependencies.size())) {
            continue;
        }

        // Get forward dependencies (successors) for current activity
        const auto& successors = RCPSPex.dependencies[i - 1]; // Convert to 0-based

        int minLateFinish = INT_MAX;
        bool hasValidSuccessor = false;

        // Find the minimum late finish time among all successors
        for (const std::string& succStr : successors) {
            // Validate string conversion
            if (succStr.empty()) continue;

            int succId = -1;
            try {
                succId = std::stoi(succStr);
            } catch (const std::exception&) {
                continue; // Skip invalid string
            }

            // Validate succId
            if (succId <= 0) continue;

            // Check if this successor already has a late finish time calculated
            auto it = latefinishTimes.find(succId);
            if (it != latefinishTimes.end()) {
                minLateFinish = std::min(minLateFinish, it->second);
                hasValidSuccessor = true;
            }
        }

        // If no valid successors found, this might be a terminal activity
        // In that case, use the project finish time
        if (!hasValidSuccessor) {
            minLateFinish = finishTime;
        }

        // Get duration for current activity
        int duration = 0;
        if (i <= static_cast<int>(size) ){
            duration = RCPSPex.activities[i-1].duration; // Convert to 0-based
        }

        // Calculate late finish time: min(successor late finish times) - duration
        latefinishTimes[i] = minLateFinish - duration;
    }

    return latefinishTimes;
}
/*
double computeSequenceLowerBoundWithMax2(
const std::vector<int>& unfinishedTransitions,
const std::vector<std::pair<int, int>>& activeTransitionIndices,
 std::map<int, int>& earlyStartTimes,
 std::map<int, int>& earlyfinishTimes,
double criticalPathEstimate,
std::map<int, int> finishedActivities

) {
  std::vector<int> path =getCriticalPath(earlyfinishTimes,RCPSPex.activities.size());
  std::map<int, int> latestartTimes=getlatefinish(RCPSPex.activities.size(),criticalPathEstimate);
    // 1. Build active set
    std::unordered_set<int> activeSet;
    for (const auto& [id, _] : activeTransitionIndices)
        activeSet.insert(id);

    // 2. Build truly unstarted list
    std::vector<int> unstartedTransitions;
    for (int id : unfinishedTransitions) {
        if (!activeSet.count(id))
            unstartedTransitions.push_back(id);
    }

    // 3. Build capacity map
    std::map<std::string, int> capacityMap;
    for (const auto& [resName, cap] : RCPSPex.resources)
        capacityMap[resName] = cap;

    // 4. Simulated resource usage timeline
    std::map<int, std::map<std::string, int>> resourceTimeline; // time -> resName -> usage

    // 5. Schedule active tasks at [0, remainingTime)
    for (const auto& [actId, remainingTime] : activeTransitionIndices) {
      path.erase(std::remove_if(path.begin(), path.end(),
          [&](int id) {
              for (const auto& [activeId, _] : activeTransitionIndices) {
                  if (id == activeId)
                    return true;
            }
            return false;
        }),
        path.end());
        const auto& act = RCPSPex.activities[actId - 1];
        for (int t = 0; t < remainingTime; ++t) {
            for (const auto& [res, demand] : act.resource_demands) {
                resourceTimeline[t][res] += demand;
            }
        }
    }



    // 6. Sort unstarted activities by descending duration
    std::vector<std::pair<int, int>> unstartedSorted; // (actId, duration)
    for (int id : unstartedTransitions) {
        int dur = RCPSPex.activities[id - 1].duration;
        unstartedSorted.emplace_back(id, dur);
    }

  path.erase(
          std::remove_if(path.begin(), path.end(),
              [&finishedActivities](int activityId) {
                  return finishedActivities.find(activityId) != finishedActivities.end();
              }
          ),
          path.end()
      );

  for (const auto& actId : path) {
    unstartedSorted.erase(
        std::remove_if(unstartedSorted.begin(), unstartedSorted.end(),
                       [&](const std::pair<int, int>& p) { return p.first == actId; }),
        unstartedSorted.end()
    );
    const auto& act = RCPSPex.activities[actId - 1];
    for (int t = earlyStartTimes[actId]; t <= act.duration+earlyStartTimes[actId]; ++t) {
      for (const auto& [res, demand] : act.resource_demands) {
        resourceTimeline[t][res] += demand;
      }
    }

  }


  // std::sort(unstartedSorted.begin(), unstartedSorted.end(),
  //             [](auto& a, auto& b) { return a.second > b.second; });

    // 7. Schedule unstarted one by one

  int maxBlockedSlotsOverall = 0;
bool valide_resoucre;
  for (const auto& [actId, duration] : unstartedSorted) {
    const auto& act = RCPSPex.activities[actId - 1];
    int est = earlyStartTimes.at(actId);
    int startTime = est;

    int minBlockedSlots = duration;
    int latefinish=latestartTimes[actId]+duration;
    // for (const auto& dep:RCPSPex.dependencies[actId - 1]) {
    //   int depId = std::stoi(dep);
    //
    //   latefinish=std::max(latefinish,earlyStartTimes[depId]);
    //   //latefinish+=duration;
    // }

   // int latefinish=earlyStartTimes[earlyStartTimes.size()];
    int counter=0;
    // Try scheduling from est onward
 //   while (true) {
      int blockedSlots = duration;

      for (int t = startTime; t <= latefinish; ++t) {
        valide_resoucre=true;
        for (const auto& [res, demand] : act.resource_demands) {
          int used = resourceTimeline[t][res];
          int available = capacityMap[res];
          if (used + demand > available) {
            blockedSlots=std::min(blockedSlots,duration-counter);
            counter=0;
            valide_resoucre=false;
            break;

          }
        }
        if (valide_resoucre){counter++;}
        if (counter==duration) {
          blockedSlots=0;
          break;
        }

      }
    blockedSlots=std::min(blockedSlots,duration-counter);

    maxBlockedSlotsOverall = std::max(maxBlockedSlotsOverall, blockedSlots);
  }


    return criticalPathEstimate+maxBlockedSlotsOverall;
}*/

double computeSequenceLowerBoundWithMax2(
const std::vector<int>& unfinishedTransitions,
const std::vector<std::pair<int, int>>& activeTransitionIndices,
 std::map<int, int>& earlyStartTimes,
 std::map<int, int>& earlyfinishTimes,
double criticalPathEstimate,
std::map<int, int> finishedActivities

) {
  std::vector<int> path =getCriticalPath(earlyfinishTimes,RCPSPex.activities.size());
  std::map<int, int> latestartTimes=getlatefinish(RCPSPex.activities.size(),criticalPathEstimate);

  path.erase(
            std::remove_if(path.begin(), path.end(),
                [&finishedActivities](int activityId) {
                    return finishedActivities.find(activityId) != finishedActivities.end();
                }
            ),
            path.end()
        );

  // 1. Build active set
    std::unordered_set<int> activeSet;
    for (const auto& [id, _] : activeTransitionIndices)
        activeSet.insert(id);

    // 2. Build truly unstarted list
    std::vector<int> unstartedTransitions;
    for (int id : unfinishedTransitions) {
      if (!activeSet.count(id) &&
     std::find(path.begin(), path.end(), id) == path.end()) {
        unstartedTransitions.push_back(id);
     }

           // unstartedTransitions.push_back(id);
    }

    // 3. Build capacity map
    std::map<std::string, int> capacityMap;
    for (const auto& [resName, cap] : RCPSPex.resources)
        capacityMap[resName] = cap;

    // 4. Simulated resource usage timeline
    std::map<int, std::map<std::string, int>> resourceTimeline; // time -> resName -> usage

    // 5. Schedule active tasks at [0, remainingTime)
    for (const auto& [actId, remainingTime] : activeTransitionIndices) {
      path.erase(std::remove_if(path.begin(), path.end(),
          [&](int id) {
              for (const auto& [activeId, _] : activeTransitionIndices) {
                  if (id == activeId)
                    return true;
            }
            return false;
        }),
        path.end());
        // const auto& act = RCPSPex.activities[actId - 1];
        // for (int t = 0; t < remainingTime; t++) {
        //     for (const auto& [res, demand] : act.resource_demands) {
        //         resourceTimeline[t][res] += demand;
        //     }
        // }
    }



    // 6. Sort unstarted activities by descending duration
    std::vector<std::pair<int, int>> unstartedSorted; // (actId, duration)
    for (int id : unstartedTransitions) {

        int dur = RCPSPex.activities[id - 1].duration;
        unstartedSorted.emplace_back(id, dur);
    }



  for (const auto& actId : path) {
    // unstartedSorted.erase(
    //     std::remove_if(unstartedSorted.begin(), unstartedSorted.end(),
    //                    [&](const std::pair<int, int>& p) { return p.first == actId; }),
    //     unstartedSorted.end()
    // );

    const auto& act = RCPSPex.activities[actId - 1];

    for (int t = earlyStartTimes[actId]; t < act.duration+earlyStartTimes[actId]; t++) {
      for (const auto& [res, demand] : act.resource_demands) {
        resourceTimeline[t][res] += demand;
      }
    }

  }
  for (auto it = unstartedSorted.begin(); it != unstartedSorted.end(); ) {
    int actId = it->first;
    bool shouldRemove = false;

    // Check if it's an active activity
    if (activeSet.count(actId)) {
      shouldRemove = true;
    }

    // Check if it's on the critical path
    if (!shouldRemove) {
      for (int pathId : path) {
        if (actId == pathId) {
          shouldRemove = true;
          break;
        }
      }
    }

    if (shouldRemove) {
      it = unstartedSorted.erase(it);
    } else {
      it++;
    }
  }

  // std::sort(unstartedSorted.begin(), unstartedSorted.end(),
  //             [](auto& a, auto& b) { return a.second > b.second; });

    // 7. Schedule unstarted one by one

  int maxBlockedSlotsOverall = 0;
bool valide_resoucre;
  for (const auto& [actId, duration] : unstartedSorted) {
    const auto& act = RCPSPex.activities[actId - 1];
    if (duration == 0) {
      continue; // Skip to next activity
    }
    int startTime = earlyStartTimes[actId];

    int latefinish=latestartTimes[actId]+duration;

    int counter=0;
    // Try scheduling from est onward
 //   while (true) {
      int blockedSlots = duration;

      for (int t = startTime; t <= latefinish; t++) {
        valide_resoucre=true;

        for (const auto& [res, demand] : act.resource_demands) {
          int used = resourceTimeline[t][res];
          int available = capacityMap[res];
          if (used + demand > available) {
            blockedSlots=std::min(blockedSlots,duration-counter);
           // counter=0;
            valide_resoucre=false;
            break;

          }
        }
        if (valide_resoucre){
          counter++;
          if (counter==duration) {
          blockedSlots=0;
          break;
        }
        }
        else {
          counter=0;
        }


      }

    blockedSlots=std::min(blockedSlots,duration-counter);
    maxBlockedSlotsOverall = std::max(maxBlockedSlotsOverall, blockedSlots);
  }


    return criticalPathEstimate+maxBlockedSlotsOverall;
}




/*
double computeSequenceLowerBoundWithMax2(
const std::vector<int>& unfinishedTransitions,
const std::vector<std::pair<int, int>>& activeTransitionIndices,
 std::map<int, int>& earlyStartTimes,
 std::map<int, int>& earlyfinishTimes,
double criticalPathEstimate,
std::map<int, int> finishedActivities

) {
  std::vector<int> path =getCriticalPath(earlyfinishTimes,RCPSPex.activities.size());
  std::map<int, int> latestartTimes=getlatefinish(RCPSPex.activities.size(),criticalPathEstimate);
    // 1. Build active set
    std::unordered_set<int> activeSet;
    for (const auto& [id, _] : activeTransitionIndices)
        activeSet.insert(id);

    // 2. Build truly unstarted list
    std::vector<int> unstartedTransitions;
    for (int id : unfinishedTransitions) {
        if (!activeSet.count(id))
            unstartedTransitions.push_back(id);
    }

    // 3. Build capacity map
    std::map<std::string, int> capacityMap;
    for (const auto& [resName, cap] : RCPSPex.resources)
        capacityMap[resName] = cap;

    // 4. Simulated resource usage timeline
    std::map<int, std::map<std::string, int>> resourceTimeline; // time -> resName -> usage

    // 5. Schedule active tasks at [0, remainingTime)
    for (const auto& [actId, remainingTime] : activeTransitionIndices) {
      path.erase(std::remove_if(path.begin(), path.end(),
          [&](int id) {
              for (const auto& [activeId, _] : activeTransitionIndices) {
                  if (id == activeId)
                    return true;
            }
            return false;
        }),
        path.end());
        const auto& act = RCPSPex.activities[actId - 1];
        for (int t = 0; t < remainingTime; ++t) {
            for (const auto& [res, demand] : act.resource_demands) {
                resourceTimeline[t][res] += demand;
            }
        }
    }



    // 6. Sort unstarted activities by descending duration
    std::vector<std::pair<int, int>> unstartedSorted; // (actId, duration)
    for (int id : unstartedTransitions) {
        int dur = RCPSPex.activities[id - 1].duration;
        unstartedSorted.emplace_back(id, dur);
    }

  path.erase(
          std::remove_if(path.begin(), path.end(),
              [&finishedActivities](int activityId) {
                  return finishedActivities.find(activityId) != finishedActivities.end();
              }
          ),
          path.end()
      );

  for (const auto& actId : path) {
    unstartedSorted.erase(
        std::remove_if(unstartedSorted.begin(), unstartedSorted.end(),
                       [&](const std::pair<int, int>& p) { return p.first == actId; }),
        unstartedSorted.end()
    );
    const auto& act = RCPSPex.activities[actId - 1];
    for (int t = earlyStartTimes[actId]; t <= act.duration+earlyStartTimes[actId]; ++t) {
      for (const auto& [res, demand] : act.resource_demands) {
        resourceTimeline[t][res] += demand;
      }
    }

  }


  // std::sort(unstartedSorted.begin(), unstartedSorted.end(),
  //             [](auto& a, auto& b) { return a.second > b.second; });

    // 7. Schedule unstarted one by one

  int maxBlockedSlotsOverall = 0;
bool valide_resoucre;
  for (const auto& [actId, duration] : unstartedSorted) {
    const auto& act = RCPSPex.activities[actId - 1];
    int est = earlyStartTimes.at(actId);
    int startTime = est;

    int minBlockedSlots = duration;
    int latefinish=latestartTimes[actId]+duration;
    // for (const auto& dep:RCPSPex.dependencies[actId - 1]) {
    //   int depId = std::stoi(dep);
    //
    //   latefinish=std::max(latefinish,earlyStartTimes[depId]);
    //   //latefinish+=duration;
    // }

   // int latefinish=earlyStartTimes[earlyStartTimes.size()];
    int counter=0;
    // Try scheduling from est onward
 //   while (true) {
      int blockedSlots = duration;

      for (int t = startTime; t <= latefinish; ++t) {
        valide_resoucre=true;
        for (const auto& [res, demand] : act.resource_demands) {
          int used = resourceTimeline[t][res];
          int available = capacityMap[res];
          if (used + demand > available) {
            blockedSlots=std::min(blockedSlots,duration-counter);
            counter=0;
            valide_resoucre=false;
            break;

          }
        }
        if (valide_resoucre){counter++;}
        if (counter==duration) {
          blockedSlots=0;
          break;
        }

      }
    blockedSlots=std::min(blockedSlots,duration-counter);

    maxBlockedSlotsOverall = std::max(maxBlockedSlotsOverall, blockedSlots);
  }


    return criticalPathEstimate+maxBlockedSlotsOverall;
}


*/
/*
double computeSequenceLowerBoundWithMax2(
const std::vector<int>& unfinishedTransitions,
const std::vector<std::pair<int, int>>& activeTransitionIndices,
 std::map<int, int>& earlyStartTimes,
 std::map<int, int>& earlyfinishTimes,
double criticalPathEstimate,
std::map<int, int> finishedActivities
) {
  std::vector<int> path =getCriticalPath(earlyfinishTimes,RCPSPex.activities.size());
  std::map<int, int> latestartTimes=getlatefinish(RCPSPex.activities.size(),criticalPathEstimate);
    // 1. Build active set
    std::unordered_set<int> activeSet;
    for (const auto& [id, _] : activeTransitionIndices)
        activeSet.insert(id);

    // 2. Build truly unstarted list
    std::vector<int> unstartedTransitions;
    for (int id : unfinishedTransitions) {
        if (!activeSet.count(id))
            unstartedTransitions.push_back(id);
    }

    // 3. Build capacity map
    std::map<std::string, int> capacityMap;
    for (const auto& [resName, cap] : RCPSPex.resources)
        capacityMap[resName] = cap;

    // 4. Simulated resource usage timeline
    std::map<int, std::map<std::string, int>> resourceTimeline; // time -> resName -> usage

    // 5. Schedule active tasks at [0, remainingTime)
    for (const auto& [actId, remainingTime] : activeTransitionIndices) {
      path.erase(std::remove_if(path.begin(), path.end(),
          [&](int id) {
              for (const auto& [activeId, _] : activeTransitionIndices) {
                  if (id == activeId)
                    return true;
            }
            return false;
        }),
        path.end());
        const auto& act = RCPSPex.activities[actId - 1];
        for (int t = 0; t < remainingTime; ++t) {
            for (const auto& [res, demand] : act.resource_demands) {
                resourceTimeline[t][res] += demand;
            }
        }
    }

    // 6. Sort unstarted activities by descending duration
    std::vector<std::pair<int, int>> unstartedSorted; // (actId, duration)
    for (int id : unstartedTransitions) {
      if (std::find(path.begin(), path.end(), id) != path.end()) {
        continue;
      }
        int dur = RCPSPex.activities[id - 1].duration;
        unstartedSorted.emplace_back(id, dur);
    }

  path.erase(
          std::remove_if(path.begin(), path.end(),
              [&finishedActivities](int activityId) {
                  return finishedActivities.find(activityId) != finishedActivities.end();
              }
          ),
          path.end()
      );

  for (const auto& actId : path) {
    unstartedSorted.erase(
        std::remove_if(unstartedSorted.begin(), unstartedSorted.end(),
                       [&](const std::pair<int, int>& p) { return p.first == actId; }),
        unstartedSorted.end()
    );
    const auto& act = RCPSPex.activities[actId - 1];
    // CRITICAL FIX: Use < instead of <=
    for (int t = earlyStartTimes[actId]; t < earlyStartTimes[actId] + act.duration; ++t) {
      for (const auto& [res, demand] : act.resource_demands) {
        resourceTimeline[t][res] += demand;
      }
    }
  }

    // 7. Schedule unstarted one by one
  int maxBlockedSlotsOverall = 0;
  bool valide_resoucre;

  for (const auto& [actId, duration] : unstartedSorted) {
    const auto& act = RCPSPex.activities[actId - 1];
    int est = earlyStartTimes.at(actId);
    int startTime = est;
    int latefinish = latestartTimes[actId] + duration;

    // Track the maximum consecutive available slots found
    int maxConsecutiveFound = 0;
    int currentConsecutive = 0;

    for (int t = startTime; t < latefinish-1; ++t) {
      valide_resoucre = true;

      for (const auto& [res, demand] : act.resource_demands) {
        int used = resourceTimeline[t][res];
        int available = capacityMap[res];
        if (used + demand > available) {
          valide_resoucre = false;
          break;
        }
      }

      if (valide_resoucre) {
        currentConsecutive++;
        maxConsecutiveFound = std::max(maxConsecutiveFound, currentConsecutive);

        // Early exit if we found enough consecutive slots
        if (currentConsecutive >= duration) {
          break;
        }
      } else {
        currentConsecutive = 0;
      }
    }

    // Calculate blocked slots: how many slots we need but couldn't find consecutively
    int blockedSlots = std::max(0, duration - maxConsecutiveFound);
    maxBlockedSlotsOverall = std::max(maxBlockedSlotsOverall, blockedSlots);
  }

    return criticalPathEstimate + maxBlockedSlotsOverall;
}
*/
double computeCoreTimeLowerBoundWithMax(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    const std::map<int, int>& earlyStartTimes,
    double criticalPathEstimate
) {
    // --- Step 1: filter active tasks out of unfinished
    std::unordered_set<int> activeSet;
    for (const auto& [id, _] : activeTransitionIndices)
        activeSet.insert(id);

    std::vector<int> unstartedTransitions;
    for (int id : unfinishedTransitions) {
        if (!activeSet.count(id))
            unstartedTransitions.push_back(id);
    }

    // --- Step 2: build capacity map and compute total work
    std::map<std::string, int> capacityMap;
    double totalWork = 0.0;
    int minCapacity = INT_MAX;

    for (const auto& [res, cap] : RCPSPex.resources) {
        capacityMap[res] = cap;
        minCapacity = std::min(minCapacity, cap);
    }

    for (int id : unstartedTransitions) {
        const auto& act = RCPSPex.activities[id - 1];
        for (const auto& [res, demand] : act.resource_demands) {
            totalWork += demand * act.duration;
        }
    }

    // --- Step 3: active task demand into timeDemand
    std::map<int, std::map<std::string, int>> timeDemand;
    int activeMaxTime = 0;

    for (const auto& [id, remaining] : activeTransitionIndices) {
        const auto& act = RCPSPex.activities[id - 1];
        for (int t = 0; t < remaining; ++t) {
            for (const auto& [res, demand] : act.resource_demands) {
                timeDemand[t][res] += demand;
            }
        }
        activeMaxTime = std::max(activeMaxTime, remaining);
    }

    // --- Step 4: binary search range
    int low = static_cast<int>(std::ceil(criticalPathEstimate));
    int high = static_cast<int>(low + std::ceil(totalWork / std::max(1, minCapacity)));
    int bestFeasible = high;

    // --- Step 5: binary search
    while (low <= high) {
        int mid = (low + high) / 2;
        bool feasible = true;

        std::map<int, std::map<std::string, int>> tempDemand = timeDemand;

        for (int id : unstartedTransitions) {
            const auto& act = RCPSPex.activities[id - 1];
            int dur = act.duration;
            int est = earlyStartTimes.at(id);
            int lst = mid - dur;

            if (lst < est) {
                feasible = false;
                break;
            }

            // Proper core interval: where the activity *must* overlap if makespan is mid
            int coreStart = std::max(est, mid - dur);
            int coreEnd = std::min(mid - 1, est + dur - 1);

            for (int t = coreStart; t <= coreEnd; ++t) {
                for (const auto& [res, demand] : act.resource_demands) {
                    tempDemand[t][res] += demand;
                    if (tempDemand[t][res] > capacityMap[res]) {
                        feasible = false;
                        break;
                    }
                }
                if (!feasible) break;
            }

            if (!feasible) break;
        }

        if (feasible) {
            bestFeasible = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    return static_cast<double>(std::max(bestFeasible, activeMaxTime));
}

double computeSequenceLowerBoundWithMax(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    const std::map<int, int>& earlyStartTimes,
    double criticalPathEstimate
) {
    // 1. Build active set
    std::unordered_set<int> activeSet;
    for (const auto& [id, _] : activeTransitionIndices)
        activeSet.insert(id);

    // 2. Build truly unstarted list
    std::vector<int> unstartedTransitions;
    for (int id : unfinishedTransitions) {
        if (!activeSet.count(id))
            unstartedTransitions.push_back(id);
    }

    // 3. Build capacity map
    std::map<std::string, int> capacityMap;
    for (const auto& [resName, cap] : RCPSPex.resources)
        capacityMap[resName] = cap;

    // 4. Simulated resource usage timeline
    std::map<int, std::map<std::string, int>> resourceTimeline; // time -> resName -> usage

    // 5. Schedule active tasks at [0, remainingTime)
    for (const auto& [actId, remainingTime] : activeTransitionIndices) {
        const auto& act = RCPSPex.activities[actId - 1];
        for (int t = 0; t < remainingTime; ++t) {
            for (const auto& [res, demand] : act.resource_demands) {
                resourceTimeline[t][res] += demand;
            }
        }
    }

    // 6. Sort unstarted activities by descending duration
    std::vector<std::pair<int, int>> unstartedSorted; // (actId, duration)
    for (int id : unstartedTransitions) {
        int dur = RCPSPex.activities[id - 1].duration;
        unstartedSorted.emplace_back(id, dur);
    }
    std::sort(unstartedSorted.begin(), unstartedSorted.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    // 7. Schedule unstarted one by one
    std::map<int, int> taskEndTimes;
    for (const auto& [actId, duration] : unstartedSorted) {
        const auto& act = RCPSPex.activities[actId - 1];
        int est = earlyStartTimes.at(actId);
        int startTime = est;

        // Try to find first time slot where it can fit
        while (true) {
            bool fits = true;

            for (int t = startTime; t < startTime + duration; ++t) {
                for (const auto& [res, demand] : act.resource_demands) {
                    int used = resourceTimeline[t][res];
                    int available = capacityMap[res];
                    if (used + demand > available) {
                        fits = false;
                        break;
                    }
                }
                if (!fits) break;
            }

            if (fits) break;
            startTime++;
        }

        // Schedule task at startTime
        for (int t = startTime; t < startTime + duration; ++t) {
            for (const auto& [res, demand] : act.resource_demands) {
                resourceTimeline[t][res] += demand;
            }
        }

        taskEndTimes[actId] = startTime + duration;
    }

    // 8. Determine last finish time
    int simulatedEnd = 0;
    for (const auto& [actId, end] : taskEndTimes)
        simulatedEnd = std::max(simulatedEnd, end);
    for (const auto& [actId, remainingTime] : activeTransitionIndices)
        simulatedEnd = std::max(simulatedEnd, remainingTime);

    return std::max(criticalPathEstimate, static_cast<double>(simulatedEnd));
}

double computeWorkloadLowerBoundWithMax(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    const std::map<int, int>& earlyStartTimes,
    double criticalPathEstimate
) {
  // Build set of active IDs
  std::unordered_set<int> activeSet;
  for (const auto& [id, _] : activeTransitionIndices)
    activeSet.insert(id);

  // Filter out truly unstarted
  std::vector<int> unstartedTransitions;
  for (int id : unfinishedTransitions) {
    if (activeSet.count(id) == 0)
      unstartedTransitions.push_back(id);
  }
   // Step 1: Estimate total horizon needed
    int project_end_est = 0;

    for (const auto& [actId, est] : earlyStartTimes) {
        int duration = RCPSPex.activities[actId - 1].duration;
        project_end_est = std::max(project_end_est, est + duration);
    }

    for (const auto& [actId, startTime] : activeTransitionIndices) {
        int duration = RCPSPex.activities[actId - 1].duration;
        project_end_est = std::max(project_end_est, startTime + duration);
    }

    // Step 2: Aggregate workload across all time units
    std::map<std::string, double> workloadPerResource;

    for (int t = 0; t < project_end_est; ++t) {
        // --- From unstarted transitions ---
        for (int actId : unstartedTransitions) {
            int est = earlyStartTimes.at(actId);
            int duration = RCPSPex.activities[actId - 1].duration;

            if (t >= est && t < est + duration) {
                const auto& activity = RCPSPex.activities[actId - 1];
                for (const auto& [resName, demand] : activity.resource_demands) {
                    workloadPerResource[resName] += demand;
                }
            }
        }

        // --- From currently active transitions ---
      for (const auto& [actId, remainingTime] : activeTransitionIndices) {
        if (t < remainingTime) { // because they started at t = 0
          const auto& activity = RCPSPex.activities[actId - 1];
          for (const auto& [resName, demand] : activity.resource_demands) {
            workloadPerResource[resName] += demand;
          }
        }
      }
    }

    // Step 3: Get resource capacities
    std::map<std::string, int> capacityMap;
    for (const auto& [resName, capacity] : RCPSPex.resources) {
        capacityMap[resName] = capacity;
    }

    // Step 4: Compute workload lower bound per resource
    int workloadBound = 0;
    for (const auto& [resName, totalWork] : workloadPerResource) {
        int cap = capacityMap[resName];
        int timeRequired = static_cast<int>(std::ceil(totalWork / cap));
        workloadBound = std::max(workloadBound, timeRequired);
    }

    // Final result
    return std::max(criticalPathEstimate, static_cast<double>(workloadBound));
    //return static_cast<double>(workloadBound);
}


double computeResourceCapacityLowerBound(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    double criticalPathEstimate
) {
  // Step 1: Build set of active IDs
  std::unordered_set<int> activeSet;
  for (const auto& [id, _] : activeTransitionIndices)
    activeSet.insert(id);

  // Step 2: Filter out active tasks → get truly unstarted
  std::vector<int> unstartedTransitions;
  for (int id : unfinishedTransitions) {
    if (!activeSet.count(id))
      unstartedTransitions.push_back(id);
  }

  // Step 3: Build capacity map
  std::map<std::string, int> capacityMap;
  for (const auto& [resName, cap] : RCPSPex.resources)
    capacityMap[resName] = cap;

  // Step 4: Accumulate workload for each resource
  std::map<std::string, double> workloadPerResource;
  for (int id : unstartedTransitions) {
    const auto& act = RCPSPex.activities[id - 1];
    for (const auto& [res, demand] : act.resource_demands) {
      workloadPerResource[res] += demand * act.duration;
    }
  }

  // Step 5: Compute LB per resource
  double lb = 0.0;
  for (const auto& [res, workload] : workloadPerResource) {
    int cap = capacityMap[res];
    if (cap > 0)
      lb = std::max(lb, std::ceil(workload / cap));
  }

  return std::max(criticalPathEstimate, lb);
}


double getBackwordsHcost(std::set<int>startedTransitions, std::vector<std::pair<int, int>>activeTransitionIndices) {
 // auto startS3 = std::chrono::high_resolution_clock::now();

  std::map<int, int> earlyfinishMap2; // Map to store activity IDs and their early finish times
  double h;

  // Iterate over started activities
  for (int activityId: startedTransitions) {
    int maxFinishTime = 0;

    // Check all backward dependencies
    for (const auto &dep: RCPSPex.backword_dependencies[activityId - 1]) {
      int depId = std::stoi(dep) - 1;

      // If dependency is in started transitions
      if (std::find(startedTransitions.begin(), startedTransitions.end(), depId + 1) != startedTransitions.end()) {
        // Find if dependency is in active transitions to get its duration
        int duration = -1;
        for (const auto &pair : activeTransitionIndices) {
          if (pair.first == depId + 1) {
            duration = pair.second;
            break;
          }
        }

        if (duration != -1) {
          // Use duration from active transitions
          maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1] + duration);
        } else {
          // Use default duration
          maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1] + RCPSPex.activities[depId].duration);
        }
      } else {
        // If dependency is not in started transitions, just use its finish time
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1]);
      }
    }

    if (std::find(startedTransitions.begin(), startedTransitions.end(), activityId + 1) != startedTransitions.end()) {
      int duration = -1;
      for (const auto &pair : activeTransitionIndices) {
        if (pair.first == activityId + 1) {
          duration = pair.second;
          break;
        }
      }

      if (duration != -1) {
        // Use duration from active transitions
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[activityId+1] + duration);
      } else {
        // Use default duration
        maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[activityId+1] + RCPSPex.activities[activityId].duration);
      }
    }
    earlyfinishMap2[activityId] = maxFinishTime;
  }

  // Find the maximum finish time
  h = 0;
  if (!earlyfinishMap2.empty()) {
    h = std::max_element(
      earlyfinishMap2.begin(),
      earlyfinishMap2.end(),
      [](const auto& p1, const auto& p2) { return p1.second < p2.second; }
    )->second;
  }

//  auto endS3 = std::chrono::high_resolution_clock::now();
  //HTIME += endS3 - startS3;
  return h;
/*

 auto startS3 = std::chrono::high_resolution_clock::now();

   std::map<int, int> earlyfinishMap2; // Map to store activity IDs and their early finish times
  //std::map<int, int> visitmap; // Map to store activity IDs and their early finish times
  double h;
  std::set<int> processedDependencies;
  // Iterate over unstarted activitiesint lastElementEarlyFinish = 0;
  //int lastElementEarlyFinish = 0;
  for (int activityId: startedTransitions) {
    int maxFinishTime = 0;
    std::set<int> processedDependencies;

    for (const auto &dep: RCPSPex.backword_dependencies[activityId - 1]) {
      int depId = std::stoi(dep) - 1;
      // if (processedDependencies.count(depId) > 0) continue;
      // processedDependencies.insert(depId);
      if (std::find(startedTransitions.begin(), startedTransitions.end(), depId + 1) != startedTransitions.end()) {
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
  h = 0;
  if (earlyfinishMap2.size()==0) {

  }
  else {
    // Find the maximum value in the map
    h = std::max_element(
      earlyfinishMap2.begin(),
      earlyfinishMap2.end(),
      [](const auto& p1, const auto& p2) { return p1.second < p2.second; }
    )->second;
  }
  // if (h != newH) {
  //   int asd;
  //   asd++;
  // }
   auto endS3 = std::chrono::high_resolution_clock::now();
   HTIME += endS3 - startS3;
  return h;
  */
}
RCPSPState::RCPSPState(): nodestatus(false) {
//  auto startS1 = std::chrono::high_resolution_clock::now();

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

 // auto endS1 = std::chrono::high_resolution_clock::now();
  //generateTIME += endS1 - startS1;

  // Change: Get indices of available transitions instead of full Transition objects
  avilableTransitionIndices = getAvilableTransitionIndices(marking);

  g = 0;
  name = 0;
}

RCPSPState_bi::RCPSPState_bi(): nodestatus(false) {
 // auto startS1 = std::chrono::high_resolution_clock::now();
  //startedActivitiys.insert(0);
  direction = true;

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
    unstartedTransitions.insert(i+1);
    //unstartedTransitions.insert(i);
  }

  // Initialize marking
  for (int i = 0; i < petri.places.size(); i++) {
    if (petri.places[i].name == initialstatename) {
      marking[petri.places[i].name] = 1;
    } else {
      marking[petri.places[i].name] = petri.places[i].state[0][0];
    }
  }

 // auto endS1 = std::chrono::high_resolution_clock::now();
  //generateTIME += endS1 - startS1;

  // Change: Get indices of available transitions instead of full Transition objects
  avilableTransitionIndices = getAvilableTransitionIndices(marking);

  avilableDeTransitionIndices = getAvilableDetransitionIndices(marking);
  g_b = 0;
  g_f = 0;
  name = 0;
  h_f=getForwardHcost(unstartedTransitions,activeTransitionIndices);
  h_b=0;
  f=h_f;
}


RCPSPState::RCPSPState(RCPSPState predecesor, Transition active, bool status, int location, uint64_t &count) {
  //auto startS4 = std::chrono::high_resolution_clock::now();

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
if (predecesor.g+predecesor.h==54) {
  int asf;
  asf++;
}


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
     // auto endS1 = std::chrono::high_resolution_clock::now();
      //generateTIME += endS1-startS4;
    }
    if (!status) {
      g += active.duration;
      finishedActivitiys[active.name] = g;

      // Remove from unstarted
      unstartedTransitions.erase(
          std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), active.name),
          unstartedTransitions.end());

      // Update durations and remove completed transitions
      // 1. נעדכן את הזמן שנותר לכולן
      for (auto& [id, remain] : activeTransitionIndices) {
        remain -= active.duration;
        if (remain < 0)
          remain = 0;
      }

      // 2. נאסוף את כל המשימות שהסתיימו
      std::vector<int> finishedNow;
      for (const auto& [id, remain] : activeTransitionIndices) {
        if (remain == 0)
          finishedNow.push_back(id);
      }

      // 3. נסיים את כל המשימות שהסתיימו
      for (int id : finishedNow) {
        finishedActivitiys[id] = g;

        for (const auto& arc : petri.Transitions[id - 1].arcs_out) {
          marking[arc.first] += arc.second;
        }

        // הסרה מ־active
        activeTransitionIndices.erase(
            std::remove_if(
                activeTransitionIndices.begin(),
                activeTransitionIndices.end(),
                [id](const std::pair<int, int>& p) { return p.first == id; }),
            activeTransitionIndices.end()
        );

        // הסרה מ־unstarted אם לא הוסרה כבר
        unstartedTransitions.erase(
            std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), id),
            unstartedTransitions.end()
        );
      }


      //auto endS1 = std::chrono::high_resolution_clock::now();
      //generateTIME += endS1-startS4;

      h=getForwardHcost(unstartedTransitions,activeTransitionIndices,finishedActivitiys);

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

      //auto endS1 = std::chrono::high_resolution_clock::now();
      //generateTIME += endS1-startS4;
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
    //  auto endS1 = std::chrono::high_resolution_clock::now();
      //generateTIME += endS1-startS4;

    }
  }
  avilableTransitionIndices = getAvilableTransitionIndices(marking);
}


RCPSPState_bi::RCPSPState_bi(RCPSPState_bi predecesor, Transition active, bool status, int location, uint64_t &count) {
 // auto startS4 = std::chrono::high_resolution_clock::now();

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
  g_b = predecesor.g_b;
  g_f = predecesor.g_f;
  h_b = predecesor.h_b;
  h_f = predecesor.h_f;


  if (direction) {
    //g_f = predecesor.g_f;

    if (status) {
      //h_f = predecesor.h_f;

      // Apply arcs_in from the transition
      for (const auto& arc : petri.Transitions[active.name-1].arcs_in) {
        marking[arc.first] -= arc.second;
      }

      // Store index and duration instead of full Transition
      activeTransitionIndices.push_back({active.name, active.duration});
      startedActivitiys.insert(active.name);
      if (active.duration==0) {
        status=0;
      }
    //  auto endS1 = std::chrono::high_resolution_clock::now();
   //   generateTIME += endS1-startS4;
    }
    if (!status) {
      g_f += active.duration;
      finishedActivitiys.insert(active.name);


      // Remove from unstarted
      unstartedTransitions.erase(active.name);

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

     // auto endS1 = std::chrono::high_resolution_clock::now();
      //generateTIME += endS1-startS4;

      h_f=getForwardHcost(unstartedTransitions,activeTransitionIndices);

    }
    f=g_f+h_f;
    h_b=getBackwardHcost2(startedActivitiys,finishedActivitiys,activeTransitionIndices);
    f=2*g_f+h_f-h_b;
    //f=2*g_f+h_f;

    avilableTransitionIndices = getAvilableTransitionIndices(marking);
  }
  else {


    // Similar transformation for the backward direction
    if (status) {
      //h_b = predecesor.h_b;

      for (const auto& arc : petri.Transitions[active.name-1].arcs_out) {
        marking[arc.first] -= arc.second;
      }

      activeTransitionIndices.push_back({active.name, 0});
      auto it = std::find(finishedActivitiys.begin(), finishedActivitiys.end(), active.name);
      if (it != finishedActivitiys.end()) {
        finishedActivitiys.erase(it);
      }
      if (active.duration==0) {
        status=0;
      }
     // auto endS1 = std::chrono::high_resolution_clock::now();
      //generateTIME += endS1-startS4;
    }
    if (!status) {
     g_b += (petri.Transitions[active.name-1].duration-active.duration);

      auto it = std::find(startedActivitiys.begin(), startedActivitiys.end(), active.name);
      if (it != startedActivitiys.end()) {
        startedActivitiys.erase(it);

      }
      unstartedTransitions.insert(active.name);

      for (int i = activeTransitionIndices.size() - 1; i >= 0; --i) {
        activeTransitionIndices[i].second += (petri.Transitions[active.name-1].duration-active.duration);
        if (activeTransitionIndices[i].second>petri.Transitions[activeTransitionIndices[i].first-1].duration) {
          activeTransitionIndices[i].second=petri.Transitions[activeTransitionIndices[i].first-1].duration;
        }
        if (activeTransitionIndices[i].first == active.name) {
          for (const auto& arc : petri.Transitions[active.name-1].arcs_in) {
            marking[arc.first] += arc.second;
          }
          activeTransitionIndices.erase(activeTransitionIndices.begin() + i);
        }
      }
      //auto endS1 = std::chrono::high_resolution_clock::now();
     // generateTIME += endS1-startS4;
      h_b=getBackwardHcost2(startedActivitiys,finishedActivitiys,activeTransitionIndices);

    }
    h_f=getForwardHcost(unstartedTransitions,activeTransitionIndices);

    avilableDeTransitionIndices = getAvilableDetransitionIndices(marking);
    f=2*g_b+h_b-h_f;
    //f=g_b;
    //f=g_b+h_b;
  }




  // You'll need to modify these functions to return indices instead of Transitions
//   if (direction) {
// }
// else {
//   }
int asdasd;
  asdasd++;
}
std::vector<int> getAvilableTransitionIndices(const std::unordered_map<std::string, int>& marking) {
  //auto startS4 = std::chrono::high_resolution_clock::now();

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
//  auto endS1 = std::chrono::high_resolution_clock::now();
  //avelableTIME += endS1-startS4;
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
  if (this->activeTransitionIndices!= other.activeTransitionIndices) {
    return false;
  }
  if (this->startedActivitiys != other.startedActivitiys) {
    return false;
  }
  // if (this->avilableTransition != other.avilableTransition) {
  //   return false;
  // }
  // if (this->activeTransitions != other.activeTransitions) {
  //   return false;
  // }
  if (this->finishedActivitiys != other.finishedActivitiys) {
    return false;
  }
  // if (this->marking != other.marking) {
  //   return false;
  // }
  return true;
}

RCPSPState_TT::RCPSPState_TT() {
  //startedActivitiys[0] = 0;

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
  for (int i = 0; i < petri.Transitions.size(); i++) {
    unstartedTransitions.push_back(i + 1);
  }

  // Initialize marking

  for (const auto& place : petri.places) {
    std::string name = place.name;
    int initialCount = place.state.empty() ? 0 : place.state[0][0];

    if (name == initialstatename) {
      // Start place has 1 token at time 0
      marking[name] = { {1, 0} };
    } else {
      if (initialCount > 0) {
        marking[name] = { {initialCount, 0} };
      } else {
        marking[name] = {}; // no token
      }
    }
  }

  // 2. Initialize resource markings (e.g., R1–R4)
  for (const auto& [resName, cap] : RCPSPex.resources) {
    if (cap > 0) {
      marking[resName] = { {cap, 0} };  // all capacity available at time 0
    } else {
      marking[resName] = {}; // no tokens
    }
  }




  // auto endS1 = std::chrono::high_resolution_clock::now();
  //generateTIME += endS1 - startS1;

  // Change: Get indices of available transitions instead of full Transition objects


  g = 0;
 avilableTransitionIndices = getAvailableTransitionIndices_TT( unstartedTransitions, finishedActivitiys, marking);

}
//
// std::vector<std::pair<int, int>> consumeResourceList(
//     const std::vector<std::pair<int, int>>& resource,
//     int amount,
//     int currentTime
// ) {
//     if (amount < 1)
//         return resource;
//
//     // Copy and sort DESCENDING by time (following Python logic)
//     std::vector<std::pair<int, int>> resourceCopy = resource;
//     std::sort(resourceCopy.begin(), resourceCopy.end(), [](const auto& a, const auto& b) {
//         return a.second > b.second; // DESCENDING by time (reverse=True in Python)
//     });
//
//     int remainingAmount = amount;
//
//     for (auto& [qty, time] : resourceCopy) {
//         // Consume from resources that are available by currentTime (time <= currentTime)
//         if (time <= currentTime && remainingAmount > 0) {
//             if (qty >= remainingAmount) {
//                 // This resource has enough to satisfy remaining demand
//                 qty -= remainingAmount;
//                 remainingAmount = 0;
//                 break;
//             } else {
//                 // Consume all of this resource and continue
//                 remainingAmount -= qty;
//                 qty = 0;
//             }
//         }
//     }
//
//     // Remove pairs with qty == 0
//     resourceCopy.erase(
//         std::remove_if(resourceCopy.begin(), resourceCopy.end(), [](const auto& p) {
//             return p.first <= 0;
//         }),
//         resourceCopy.end()
//     );
//
//     return resourceCopy;
// }
std::vector<std::pair<int, int>> consumeResourceList(
    const std::vector<std::pair<int, int>>& resource,
    int amount,
    int currentTime
) {
  if (amount < 1)
    return resource;

  // Check if already sorted to avoid unnecessary sorting
  std::vector<std::pair<int, int>> resourceCopy = resource;

  // Only sort if not already sorted (you could maintain sorted invariant)
  std::sort(resourceCopy.begin(), resourceCopy.end(), [](const auto& a, const auto& b) {
      return a.second > b.second; // DESCENDING by time
  });

  int remainingAmount = amount;

  for (auto& [qty, time] : resourceCopy) {
    if (time <= currentTime && remainingAmount > 0) {
      if (qty >= remainingAmount) {
        qty -= remainingAmount;
        remainingAmount = 0;
        break;
      } else {
        remainingAmount -= qty;
        qty = 0;
      }
    }
  }

  // Use erase-remove idiom efficiently
  resourceCopy.erase(
      std::remove_if(resourceCopy.begin(), resourceCopy.end(),
                    [](const auto& p) { return p.first <= 0; }),
      resourceCopy.end()
  );

  return resourceCopy;
}
std::vector<std::pair<int, int>> return_resource(
    const std::vector<std::pair<int, int>>& resource,
    int amount,
    int return_time) {

  // Create a copy of the input resource vector
  std::vector<std::pair<int, int>> resource_copy = resource;

  // If amount is less than 1, just return the copy without changes
  if (amount < 1) {
    return resource_copy;
  }

  // Check if there is already an entry with the return_time
  bool found = false;
  for (auto& item : resource_copy) {
    if (item.second == return_time) {
      // Found an existing entry with the same return time, add to it
      item.first += amount;
      found = true;
      break;
    }
  }

  // If no entry with the return_time was found, create a new one
  if (!found) {
    resource_copy.push_back({amount, return_time});
  }

  return resource_copy;
}

double computeEarliestFinish(int activityId,
                             std::map<int, double>& earlyFinishMemo,
                             const std::vector<int>& unstartedTransitions,
                             const std::map<int, int>& finishedActivities) {
    // Memoization check
    if (earlyFinishMemo.count(activityId)) return earlyFinishMemo[activityId];

    // If activity is in finishedActivities, its finish time is 0
    if (finishedActivities.count(activityId)) {
        earlyFinishMemo[activityId] = 0.0;
        return 0.0;
    }

    int internalId = activityId - 1;
    double maxFinish = 0.0;

    for (const std::string& dep : RCPSPex.backword_dependencies[internalId]) {
        int depId = std::stoi(dep); // 1-based
        int depInternal = depId - 1;

        // Recurse only if it's in unstartedTransitions
        if (std::find(unstartedTransitions.begin(), unstartedTransitions.end(), depId) != unstartedTransitions.end()) {
            double depFinish = computeEarliestFinish(depId, earlyFinishMemo, unstartedTransitions, finishedActivities)
                             + RCPSPex.activities[depInternal].duration;
            maxFinish = std::max(maxFinish, depFinish);
        } else {
            // If dependency is finished, consider its finish time as stored in earlyFinishMemo
            maxFinish = std::max(maxFinish, earlyFinishMemo[depId]);
        }
    }

    earlyFinishMemo[activityId] = maxFinish;
    return maxFinish;
}

//maby finsihed activity is diffrent with indipendent
double getForwardHcost_TT(std::vector<int>unstartedTransitions, std::map<int, int> finishedActivitiys
  //,std::string mode="defult"                     // activityID -> finish time
) {
  //auto startS3 = std::chrono::high_resolution_clock::now();
  //std::vector<std::pair<int, int>> activeTransitionIndices;

   std::map<int, int> earlyfinishMap2; // Map to store activity IDs and their early finish times
   //std::map<int, int> earlyfinishMap3; // Map to store activity IDs and their early finish times
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

          maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1] + RCPSPex.activities[depId].duration);
       //   earlyfinishMap3[depId+1] = earlyfinishMap2[depId+1] + RCPSPex.activities[depId].duration;

      }
      else {
         maxFinishTime = std::max(maxFinishTime, earlyfinishMap2[depId+1]);
       //  earlyfinishMap3[depId+1] = earlyfinishMap2[depId+1];
    //    maxFinishTime = std::max(maxFinishTime, 0);
        //earlyfinishMap3[depId+1] = 0;
      }
    }

    earlyfinishMap2[activityId] = maxFinishTime;
    //earlyfinishMap3[activityId] = maxFinishTime;
    //std::cout <<activityId<<":"<< earlyfinishMap[activityId]+RCPSPex.activities[activityId-1].duration << std::endl;
    // For last element with duration 0, just use the max finish time of dependencies
  }
  if (earlyfinishMap2.size()==0) {
    h = 0;
  }
  else {
    h = earlyfinishMap2.rbegin()->second;;

  }
  // if (mode == "cs") {
  //   return computeSequenceLowerBoundWithMax2(
  //       unstartedTransitions,
  //       activeTransitionIndices,
  //       earlyfinishMap2,
  //       earlyfinishMap3,
  //       h,
  //       finishedActivitiys); // BL_Cs heuristic
  // }
  // else {
    return h;
 // }
 // return h;
 // return std::max(computeResourceCapacityLowerBound(unstartedTransitions,activeTransitionIndices,h), computeSequenceLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h));//BL_RC huristic
  //return computeResourceCapacityLowerBound(unstartedTransitions,activeTransitionIndices,h);//BL_Cs huristic
  //return computeSequenceLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_Cs huristic
  //return computeSequenceLowerBoundWithMax2(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,earlyfinishMap3,h,finishedActivitiys);//BL_Cs huristic
  //return computeCoreTimeLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_CT huristic
  //return computeWorkloadLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_CC huristic
///
 //return h;

}

RCPSPState_TT::RCPSPState_TT(const RCPSPState_TT &prev, int transitionId, int firingTime) {
    // 1. Copy structures from previous state
    startedActivitiys = prev.startedActivitiys;
    finishedActivitiys = prev.finishedActivitiys;
    unstartedTransitions = prev.unstartedTransitions;
    marking = prev.marking;
  // if (prev.g+prev.h>63) {
  //   int asf;
  //   asf++;
  // }
    // 2. Cache frequently accessed data
    const Transition& transition = petri.Transitions[transitionId - 1];
    const Activity& act = RCPSPex.activities[transitionId - 1];
    const int duration = act.duration;
    const int activityFinishTime = firingTime + duration;

    // 3. Update Petri net marking
    for (const auto& [place, outAmount] : transition.arcs_out) {
        if (marking.count(place) == 0) {
            marking[place] = {{outAmount, firingTime + transition.duration}};
        } else {
            marking[place].emplace_back(outAmount, firingTime + transition.duration);
        }
    }

    // 4. Update activity states
    startedActivitiys[transitionId] = firingTime;
    finishedActivitiys[transitionId] = activityFinishTime;

    // 5. Consume resources - optimized loop
    for (const std::string& res : resourceNames) {
        auto demandIt = act.resource_demands.find(res);
        if (demandIt != act.resource_demands.end() && demandIt->second > 0) {
            marking[res] = consumeResourceList(marking[res], demandIt->second, firingTime);
            // Uncomment when ready: marking[res] = return_resource(marking[res], demandIt->second, activityFinishTime);
        }
    }

    // 6. Remove from unstarted - optimized removal
    auto it = std::find(unstartedTransitions.begin(), unstartedTransitions.end(), transitionId);
    if (it != unstartedTransitions.end()) {
        unstartedTransitions.erase(it); // Single iterator erase is faster
    }

    // 7. Calculate g-score efficiently
    if (finishedActivitiys.empty()) {
        g = 0;
    }
   else {
        for (const auto& [id, finishTime] : finishedActivitiys) {
            if (finishTime > g) {
                g = finishTime;
            }
        }
    }

    // 8. REQUIRED: Calculate available transitions
    avilableTransitionIndices = getAvailableTransitionIndices_TT(unstartedTransitions, finishedActivitiys, marking);

int lastActivityId = -1;
  int maxTime = -1;

  // Find last finished activity by ID instead of name
  for (const auto& [id, time] : finishedActivitiys) {
    if (time > maxTime) {
      maxTime = time;
      lastActivityId = id;
    }
  }

  if (lastActivityId != -1) {
    const std::string& lastActivityName = RCPSPex.activities[lastActivityId - 1].name;

    // Pre-reserve vectors
    std::vector<int> independentSet;
    independentSet.reserve(unstartedTransitions.size());

    // Filter independent transitions
    for (int actIdx : unstartedTransitions) {
      const std::string& actName = RCPSPex.activities[actIdx - 1].name;
      if (RCPSPex.deep_dependencies.find({lastActivityName, actName}) == RCPSPex.deep_dependencies.end()) {
        independentSet.push_back(actIdx);
      }
    }

    // Create lookup set for efficient filtering
    std::unordered_set<int> independentLookup(independentSet.begin(), independentSet.end());
    std::vector<int> newUnstartedTransitions;
    newUnstartedTransitions.reserve(unstartedTransitions.size());

    for (int id : unstartedTransitions) {
      if (independentLookup.find(id) == independentLookup.end()) {
        newUnstartedTransitions.push_back(id);
      }
    }

    // 10. Calculate heuristic efficiently
    int latestStart = 0;
    for (const auto& [id, startTime] : startedActivitiys) {
      if (startTime > latestStart) {
        latestStart = startTime;
      }
    }

    int unkTime = g - latestStart;
    std::map<int, int> finishedActivitiysnew=finishedActivitiys;                         // activityID -> finish time
    for (int actIdx : independentSet) {
      finishedActivitiysnew[actIdx] = 0;
    }

  // getForwardHcost_TT(unstartedTransitions,finishedActivitiys) - unkTime;
      h= std::max(getForwardHcost_TT(unstartedTransitions,finishedActivitiys) - unkTime,
                 getForwardHcost_TT(newUnstartedTransitions,finishedActivitiysnew));
  } else {
    // Fallback if no finished activities
    h= getForwardHcost_TT(unstartedTransitions,finishedActivitiys);
  }
}
//
// RCPSPState_TT::RCPSPState_TT(const RCPSPState_TT &prev,int transitionId, int firingTime) {
//   // העתקת מבנים מהמצב הקודם
//   startedActivitiys = prev.startedActivitiys;
//   finishedActivitiys = prev.finishedActivitiys;
//   unstartedTransitions = prev.unstartedTransitions;
//   marking = prev.marking;  // כולל עותק עמוק של ה־vectors בפנים
//
//   //int firingTime = 0;
//   const Transition& transition = petri.Transitions[transitionId - 1];
//
//   // Compute firing time (max of input places' times)
//
//   for (const auto& [place, outAmount] : transition.arcs_out) {
//     // Add new tokens with updated time
//     if (marking.count(place) == 0) {
//       // If place doesn't exist in marking yet, create it
//       marking[place] = {{outAmount, firingTime + transition.duration}};
//     } else {
//       // If place exists, add a new token with the specified amount and time
//       marking[place].push_back({outAmount, firingTime + transition.duration});
//     }
//   }
//
//       const Activity& act = RCPSPex.activities[transitionId - 1];
//
//       int duration = act.duration;
//
//       startedActivitiys[transitionId] = firingTime;
//       finishedActivitiys[transitionId] = firingTime + duration;
//       //
//       //  2. צריכת משאבים (marking)
//
//       for (const std::string& res : resourceNames) {
//         int demand = 0;
//
//         // Check if this activity demands this resource
//         if (act.resource_demands.count(res)) {
//           demand = act.resource_demands.at(res);
//         }
//
//         if (demand > 0) {
//           marking[res] = consumeResourceList(marking[res], demand, firingTime);
//
//         }
//       }
//
//
//
//       // 3. עדכון unstarted
//       unstartedTransitions.erase(
//           std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), transitionId),
//           unstartedTransitions.end()
//       );
//
//       // 4. נעדכן זמינות (אם צריך)
//       if (startedActivitiys.empty()) {
//         g=0;
//       }
//       else {
//         auto maxIt = std::max_element(
//           finishedActivitiys.begin(), finishedActivitiys.end(),
//           [](const std::pair<const int, int>& a, const std::pair<const int, int>& b) {
//               return a.second < b.second;
//           });
//         g=maxIt->second;
//       }
//
//   avilableTransitionIndices = getAvailableTransitionIndices_TT(unstartedTransitions,finishedActivitiys,marking);
//   std::string last_activity;
//   int max_time = -1;
//
//   // Find the activity with the latest finish time
//   for (const auto& [name, time] : finishedActivitiys) {
//     if (time > max_time) {
//       max_time = time;
// last_activity = RCPSPex.activities[name-1].name;    }
//   }
//
//   std::vector<int> independentSet;
//
//   // Filter independent transitions (not dependent on last finished)
//   for (int actIdx : unstartedTransitions) {
//     const std::string& actName = RCPSPex.activities[actIdx-1].name;
//     if (RCPSPex.deep_dependencies.find({last_activity, actName}) == RCPSPex.deep_dependencies.end()) {
//       independentSet.push_back(actIdx);
//     }
//   }
//
//   // Remove independent transitions from unstartedTransitions
//   std::unordered_set<int> independentLookup(independentSet.begin(), independentSet.end());
//   std::vector<int> newUnstartedTransitions;
//   for (int id : unstartedTransitions) {
//     if (independentLookup.find(id) == independentLookup.end()) {
//       newUnstartedTransitions.push_back(id);
//     }
//   }
//
//   // Compute unknown time
//   int latest_start = 0;
//   for (const auto& [id, start_time] : startedActivitiys) {
//     if (start_time > latest_start) {
//       latest_start = start_time;
//     }
//   }
//   int unk_time = g - latest_start;
//   //h=getForwardHcost_TT(unstartedTransitions,finishedActivitiys)-unk_time;
//       h=std::max(getForwardHcost_TT(unstartedTransitions)-unk_time,getForwardHcost_TT(newUnstartedTransitions));
//  // h=getForwardHcost_TT(unstartedTransitions,finishedActivitiys);
//   }

/*
std::vector<std::pair<int, int>> RCPSPState_TT::getAvailableTransitionIndices_TT(
    const std::vector<int> &unstartedTransitions,
    const std::map<int, int> &finishedActivities,
    const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &marking,
    int currentTime) {
  std::vector<std::pair<int, int>> available; // (transition ID, earliest firing time)

  for (int transId : unstartedTransitions) {
    const Activity& act = RCPSPex.activities[transId - 1];

    // 1. Check precedence constraints
    bool allDependenciesFinished = true;
    for (const std::string& predStr : RCPSPex.backword_dependencies[transId - 1]) {
      int predId = std::stoi(predStr);
      if (finishedActivities.find(predId) == finishedActivities.end()) {
        allDependenciesFinished = false;
        break;
      }
    }
    if (!allDependenciesFinished)
      continue;

    // 2. Find earliest time when enough resources are available
    int earliestFire = currentTime;
    for (const auto& [res, demand] : act.resource_demands) {
      auto it = marking.find(res);
      if (it == marking.end()) {
        earliestFire = -1;
        break;
      }

      int availableAmount = it->second.count("count") ? it->second.at("count") : 0;
      int availableTime = it->second.count("time") ? it->second.at("time") : 0;

      if (availableAmount < demand) {
        earliestFire = -1;
        break;
      }

      earliestFire = std::max(earliestFire, availableTime);
    }

    if (earliestFire != -1)
      available.emplace_back(transId, earliestFire);
  }

  return available;
}
*/
/*
std::vector<std::pair<int, int>> RCPSPState_TT::getAvailableTransitionIndices_TT(
    const std::vector<int> &unstartedTransitions,
    const std::map<int, int> &finishedActivities,
    const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &marking
    ) {
 std::vector<std::pair<int, int>> available; // (transition ID, earliest firing time)

    for (int transId : unstartedTransitions) {
        const Activity& act = RCPSPex.activities[transId - 1];

        // 1. Check precedence constraints
        bool allDependenciesFinished = true;
        int maxPredFinishTime = 0;

        for (const std::string& predStr : RCPSPex.backword_dependencies[transId - 1]) {
            int predId = std::stoi(predStr);
            auto it = finishedActivities.find(predId);
            if (it == finishedActivities.end()) {
                allDependenciesFinished = false;
                break;
            } else {
                maxPredFinishTime = std::max(maxPredFinishTime, it->second);
            }
        }

        if (!allDependenciesFinished)
            continue;

        // 2. Check resource availability and find max resource ready time
        bool resourcesAvailable = true;
        int maxResourceReadyTime = maxPredFinishTime; // Start from earliest possible time

        for (const auto& [res, demand] : act.resource_demands) {
            auto it = marking.find(res);
            if (it == marking.end()) {
                resourcesAvailable = false;
                break;
            }

            // Sort tokens by time
            auto resourceTokens = it->second; // Copy for sorting
            std::sort(resourceTokens.begin(), resourceTokens.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });

            int totalAvailable = 0;
            int resourceReadyTime = -1;

            // First, accumulate all resources available at or before maxPredFinishTime
            for (const auto& [amount, time] : resourceTokens) {
                if (time <= maxPredFinishTime) {
                    totalAvailable += amount;
                }
            }

            // Check if we already have enough at maxPredFinishTime
            if (totalAvailable >= demand) {
                resourceReadyTime = maxPredFinishTime;
            } else {
                // Need to wait for more resources that become available after maxPredFinishTime
                for (const auto& [amount, time] : resourceTokens) {
                    if (time > maxPredFinishTime) {
                        totalAvailable += amount;
                        if (totalAvailable >= demand) {
                            resourceReadyTime = time;
                            break;
                        }
                    }
                }
            }

            if (resourceReadyTime == -1) {
                resourcesAvailable = false;
                break;
            }

            maxResourceReadyTime = std::max(maxResourceReadyTime, resourceReadyTime);
        }

        if (resourcesAvailable) {
            int finalEarliestFire = std::max(maxPredFinishTime, maxResourceReadyTime);
            available.emplace_back(transId, finalEarliestFire);
        }
    }

    return available;
}
*/

std::vector<std::pair<int, int>> getAvailableTransitionIndices_TT(
    const std::vector<int> &unstartedTransitions,
    const std::map<int, int> &finishedActivities,
    const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &marking
) {
    std::vector<std::pair<int, int>> available;

    for (int transId : unstartedTransitions) {
        const Activity &act = RCPSPex.activities[transId - 1];

        // 1. Precedence constraints
        bool allPredsFinished = true;
        int maxPredFinishTime = 0;

        for (const std::string &predStr : RCPSPex.backword_dependencies[transId - 1]) {
            int predId = std::stoi(predStr);
            auto it = finishedActivities.find(predId);
            if (it == finishedActivities.end()) {
                allPredsFinished = false;
                break;
            } else {
                maxPredFinishTime = std::max(maxPredFinishTime, it->second);
            }
        }

        if (!allPredsFinished)
            continue;

        // 2. Resource availability
        bool resourcesOK = true;
        int maxResourceTime = maxPredFinishTime;

        for (const auto &[res, demand] : act.resource_demands) {
            auto it = marking.find(res);
            if (it == marking.end()) {
                resourcesOK = false;
                break;
            }

            auto tokens = it->second;
            std::sort(tokens.begin(), tokens.end(),
                      [](auto &a, auto &b) { return a.second < b.second; });

            int totalAvailable = 0;
            int resourceReadyTime = -1;

            for (const auto &[amt, time] : tokens) {
                if (time <= maxPredFinishTime) {
                    totalAvailable += amt;
                }
            }

            if (totalAvailable >= demand) {
                resourceReadyTime = maxPredFinishTime;
            } else {
                for (const auto &[amt, time] : tokens) {
                    if (time > maxPredFinishTime) {
                        totalAvailable += amt;
                        if (totalAvailable >= demand) {
                            resourceReadyTime = time;
                            break;
                        }
                    }
                }
            }

            if (resourceReadyTime == -1) {
                resourcesOK = false;
                break;
            }

            maxResourceTime = std::max(maxResourceTime, resourceReadyTime);
        }

        if (resourcesOK) {
            available.emplace_back(transId, maxResourceTime);
        }
    }

    return available;
}


//
// std::vector<int> RCPSPState_TT::getOptionalTransitions_TT(
//     const std::vector<int> &unstartedTransitions,
//     const std::map<int, int> &startedActivities  // Note: started, not finished
// ) {
//     std::vector<int> optionalTransitions;
//
//     // Convert started activities keys to set for faster lookup
//     std::unordered_set<int> startedActivitiesKeys;
//     for (const auto &pair : startedActivities) {
//         startedActivitiesKeys.insert(pair.first);
//     }
//
//     for (int transId : unstartedTransitions) {
//         // Check if any of this transition's dependencies have been started
//         const std::vector<std::string> &dependencies = RCPSPex.backword_dependencies[transId - 1];
//
//         bool hasDependencyStarted = false;
//         for (const std::string &depStr : dependencies) {
//             int depId = std::stoi(depStr);
//             if (startedActivitiesKeys.find(depId) != startedActivitiesKeys.end()) {
//                 hasDependencyStarted = true;
//                 break;
//             }
//         }
//
//         if (hasDependencyStarted) {
//             optionalTransitions.push_back(transId);
//         }
//     }
//
//     return optionalTransitions;
// }
//
// // Second function: Check available transitions from the optional ones
// std::vector<std::pair<int, int>> RCPSPState_TT::checkAvailableTransitions_TT(
//     const std::vector<int> &optionalTransitions,
//     const std::map<int, int> &finishedActivities,
//     const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &marking
// ) {
//     std::vector<std::pair<int, int>> available;
//
//     for (int transId : optionalTransitions) {
//         const Activity &act = RCPSPex.activities[transId - 1];
//
//         // 1. Precedence constraints - check if ALL predecessors are finished
//         bool allPredsFinished = true;
//         int maxPredFinishTime = 0;
//
//         for (const std::string &predStr : RCPSPex.backword_dependencies[transId - 1]) {
//             int predId = std::stoi(predStr);
//             auto it = finishedActivities.find(predId);
//             if (it == finishedActivities.end()) {
//                 allPredsFinished = false;
//                 break;
//             } else {
//                 maxPredFinishTime = std::max(maxPredFinishTime, it->second);
//             }
//         }
//
//         if (!allPredsFinished)
//             continue;
//
//         // 2. Resource availability
//         bool resourcesOK = true;
//         int maxResourceTime = maxPredFinishTime;
//
//         for (const auto &[res, demand] : act.resource_demands) {
//             auto it = marking.find(res);
//             if (it == marking.end()) {
//                 resourcesOK = false;
//                 break;
//             }
//
//             auto tokens = it->second;
//             std::sort(tokens.begin(), tokens.end(),
//                       [](const auto &a, const auto &b) { return a.second < b.second; });
//
//             int totalAvailable = 0;
//             int resourceReadyTime = -1;
//
//             // Count resources available at maxPredFinishTime
//             for (const auto &[amt, time] : tokens) {
//                 if (time <= maxPredFinishTime) {
//                     totalAvailable += amt;
//                 }
//             }
//
//             if (totalAvailable >= demand) {
//                 resourceReadyTime = maxPredFinishTime;
//             } else {
//                 // Look for future resource availability
//                 for (const auto &[amt, time] : tokens) {
//                     if (time > maxPredFinishTime) {
//                         totalAvailable += amt;
//                         if (totalAvailable >= demand) {
//                             resourceReadyTime = time;
//                             break;
//                         }
//                     }
//                 }
//             }
//
//             if (resourceReadyTime == -1) {
//                 resourcesOK = false;
//                 break;
//             }
//
//             maxResourceTime = std::max(maxResourceTime, resourceReadyTime);
//         }
//
//         if (resourcesOK) {
//             available.emplace_back(transId, maxResourceTime);
//         }
//     }
//
//     return available;
// }







// std::vector<std::pair<int, int>> RCPSPState_TT::getAvailableTransitionIndices_TT_old(
//     const std::vector<int> &unstartedTransitions,
//     const std::map<int, int> &finishedActivities,
//     const std::unordered_map<std::string, std::vector<std::pair<int, int>>> &marking
//     ) {
//
//  std::vector<std::pair<int, int>> available; // (transition ID, earliest firing time)
//
//     for (int transId : unstartedTransitions) {
//         const Activity& act = RCPSPex.activities[transId - 1];
//
//         // 1. Check precedence constraints
//         bool allDependenciesFinished = true;
//         for (const std::string& predStr : RCPSPex.backword_dependencies[transId - 1]) {
//             int predId = std::stoi(predStr);
//             if (finishedActivities.find(predId) == finishedActivities.end()) {
//                 allDependenciesFinished = false;
//                 break;
//             }
//         }
//         if (!allDependenciesFinished)
//             continue;
//
//         // 2. Find earliest time when ALL required backword_dependencies are available
//         int earliestFire = 0;
//       for (const std::string& predStr : RCPSPex.backword_dependencies[transId - 1]) {
//         int predId = std::stoi(predStr);
//         if (finishedActivities.find(predId) != finishedActivities.end()) {
//           earliestFire = std::max(earliestFire, finishedActivities.at(predId));
//         }
//       }
//
//
//
//
//
//         bool resourcesAvailable = true;
//
//         for (const auto& [res, demand] : act.resource_demands) {
//             auto it = marking.find(res);
//             if (it == marking.end()) {
//                 resourcesAvailable = false;
//                 break;
//             }
//
//             // Sort tokens by time to process them in order
//             auto resourceTokens = it->second; // Make a copy to sort
//             std::sort(resourceTokens.begin(), resourceTokens.end(),
//                      [](const auto& a, const auto& b) { return a.second < b.second; });
//
//             // Find the earliest time when we have enough of THIS resource
//             int totalAvailable = 0;
//             int resourceReadyTime = -1;
//
//             for (const auto& [amount, time] : resourceTokens) {
//                 if (totalAvailable < demand) {
//                     // We still need more resources
//                     totalAvailable += amount;
//                     if (totalAvailable >= demand) {
//                         // This is the moment we first have enough - use this time
//                         resourceReadyTime = time;
//                         break;
//                     }
//                 }
//             }
//
//             if (resourceReadyTime == -1) {
//                 // Not enough of this resource available
//                 resourcesAvailable = false;
//                 break;
//             }
//
//             // The earliest firing time is when ALL resources are ready
//             // So we take the maximum (latest) time among all resource types
//             earliestFire = std::max(earliestFire, resourceReadyTime);
//         }
//
//         if (resourcesAvailable)
//             available.emplace_back(transId, earliestFire);
//     }
//
//     return available;
//
// }



bool RCPSPState_TT::operator==(const RCPSPState_TT &other) const {
  return true;
}




