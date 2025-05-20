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

double computeSequenceLowerBoundWithMax(
    const std::vector<int>& unfinishedTransitions,
    const std::vector<std::pair<int, int>>& activeTransitionIndices,
    const std::map<int, int>& earlyStartTimes,
    double criticalPathEstimate
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



double getForwardHcost(std::vector<int>unstartedTransitions, std::vector<std::pair<int, int>>activeTransitionIndices) {
  //auto startS3 = std::chrono::high_resolution_clock::now();

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
///
// return h;
 // return std::max(computeResourceCapacityLowerBound(unstartedTransitions,activeTransitionIndices,h), computeSequenceLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h));//BL_RC huristic
  //return computeResourceCapacityLowerBound(unstartedTransitions,activeTransitionIndices,h);//BL_Cs huristic
  return computeSequenceLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_Cs huristic
  //return computeCoreTimeLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_CT huristic
  //return computeWorkloadLowerBoundWithMax(unstartedTransitions,activeTransitionIndices,earlyfinishMap2,h);//BL_CC huristic
///
 return h;

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

      //auto endS1 = std::chrono::high_resolution_clock::now();
      //generateTIME += endS1-startS4;

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



  // You'll need to modify these functions to return indices instead of Transitions
//   if (direction) {
// }
// else {
//   }
int asdasd;
  asdasd++;
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
    //h_b=getBackwardHcost2(startedActivitiys,finishedActivitiys,activeTransitionIndices);
    //f=2*g_f+h_f-h_b;
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
    f=g_b+h_b;
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

std::vector<int> RCPSPState_TT::getAvailableTransitionIndices_TT() const {
  std::vector<int> available;

  for (int id : unstartedTransitions) {
    const Activity& activity = RCPSPex.activities[id - 1];

    // 1. קדימויות: כל הקודמים חייבים להיות ב-finished
    bool depsMet = true;
    for (const std::string& predStr : RCPSPex.backword_dependencies[id - 1]) {
      int predId = std::stoi(predStr);
      if (finishedActivitiys.count(predId) == 0) {
        depsMet = false;
        break;
      }
    }
    if (!depsMet) continue;

    // 2. משאבים זמינים
    bool resourcesAvailable = true;
    for (const auto& [resName, amount] : activity.resource_demands) {
      auto it = marking.find(resName);
      if (it == marking.end() || it->second < amount) {
        resourcesAvailable = false;
        break;
      }
    }
    if (!resourcesAvailable) continue;

    // אם עברנו את שני התנאים
    available.push_back(id);
  }

  return available;
}


