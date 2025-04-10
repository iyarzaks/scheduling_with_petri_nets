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
//    getPetri(petri);
//    getRCPSP(RCPSPex);
//    RCPSPState first;
//    // std::cout<<petri.Transitions[0].name<<std::endl;
//    std::vector<RCPSPState> network;
//    network.emplace_back(first);
//    // auto start_time = std::chrono::high_resolution_clock::now(); // Start the timer
//
//   //int i;
//   int count=0;
//    // for (int j=0;j<network.size();j++) {
//    //   std::cout<<"Node:"<<network[j].name<<std::endl;
//    //   std::cout<<"Avilable Transitions:";
//    //   for (int k=0;k<network[j].avilableTransition.size();k++) {std::cout<<network[j].avilableTransition[k].name<<" ";}
//    // }
//    // std::atomic<int> counter = 0;
//    int i = 0;
//
//    // Launch the size printing thread
//    //std::thread printer(printNetworkSize, std::cref(network));
//
//   while (true) {
//
//
//
//     if (network[i].expanded==0) {
//       network[i].expanded=1;
//       GetNabor(network,i,count);
//     }
//       // for (int j=0;j<network.size();j++) {
//       //   if (network[j].expanded==0) {
//       //     std::cout<<"Node:"<<network[j].name<<std::endl;
//       //     std::cout<<"with g of:"<<network[j].g<<std::endl;
//       //     std::cout<<"activeTransitions:"<<std::endl;
//       //     for (int k=0;k<network[j].activeTransitions.size();k++) {
//       //       std::cout<<network[j].activeTransitions[k].name<<" ";
//       //     }
//       //     std::cout<<std::endl;
//       //     std::cout<<"Avilable Transitions:"<<std::endl;
//       //     for (int k=0;k<network[j].avilableTransition.size();k++) {
//       //       std::cout<<network[j].avilableTransition[k].name<<" ";
//       //     }
//       //     //std::cout<<std::endl;
//       //   }
//       // }
// for (int j=0;j<network.size();j++) {
//   if (network[j].marking[finalstatename]==1) {
// //std::cout<<"finish with total time of:"<<network[j].g;
//     auto end_time = std::chrono::high_resolution_clock::now(); // End the timer
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//     std::cout << "Execution time: " << duration << " ms" << std::endl;
//     std::cout << "Current network size: " << network.size() << std::endl;
//     return 1;
//   }
// }
// int f=1000;
// for (int j=0;j<network.size();j++) {
//   if (network[j].expanded==0) {
//     if (network[j].g+network[j].h<=f) {
//       f=network[j].g+network[j].h;
//       i=j;
//     }
//   }
// }
//      //std::cout<<"-----------------"<<std::endl;
//  //std::cout<<"with f of:"<<f<<std::endl;
//  //std::cout<<"expanding:"<<std::endl;
//
//   }
}

// int ChooseExpansion(std::vector<RCPSPState> network) {
// return 1;
// }

//SearchGraph search_graph;

//probebly Very inefficent
/*
std::vector<Transition> getAvilableTransitions(std::map<std::string, int> marking) {
   auto startS1 = std::chrono::high_resolution_clock::now();

  std::vector<Transition> avilableTransitions;
  for (int i=0;i<petri.Transitions.size();i++) {
    int avilable=0;
    int requirment=0;

    for (const auto& arc : petri.Transitions[i].arcs_in) {
      if (marking[arc.first]>=1){avilable+=std::min(marking[arc.first],arc.second);}
      requirment+=arc.second;
    }
    if (avilable>=requirment) {
      avilableTransitions.push_back(petri.Transitions[i]);
    }

  }
   auto endS1 = std::chrono::high_resolution_clock::now();

   avelableTIME += endS1-startS1;
  return avilableTransitions;
}
//*/
//std::vector<Transition> getAvilableTransitions(const std::unordered_map<std::string, int>& marking) {
//   auto startS1 = std::chrono::high_resolution_clock::now();
//
//   std::vector<Transition> avilableTransitions;
//   avilableTransitions.reserve(petri.Transitions.size());  // Reserve memory to avoid multiple reallocations
//
//   std::cout << "Current marking: ";
//   for (const auto& m : marking) {
//     std::cout << m.first << ":" << m.second << " ";
//   }
//   std::cout << std::endl;
//
//   for (const auto& transition : petri.Transitions) {
//     std::cout << "Checking transition " << transition.name << std::endl;
//     int avilable = 0, requirment = 0;
//     bool canFire = true;
//
//     for (const auto& arc : transition.arcs_in) {
//       auto it = marking.find(arc.first);
//       int tokenCount = (it != marking.end()) ? it->second : 0;
//
//       std::cout << "  Place " << arc.first << " has " << tokenCount << " tokens, needs " << arc.second << std::endl;
//
//       if (tokenCount < arc.second) {
//         canFire = false;  // Not enough tokens to fire
//         std::cout << "  Cannot fire: insufficient tokens" << std::endl;
//         break;            // Stop checking further
//       }
//       avilable += std::min(tokenCount, arc.second);
//       requirment += arc.second;
//     }
//
//     if (canFire) {
//       std::cout << "  Transition " << transition.name << " can fire!" << std::endl;
//       avilableTransitions.push_back(transition);
//     }
//   }
//
//   std::cout << "Found " << avilableTransitions.size() << " available transitions" << std::endl;
//
//   auto endS1 = std::chrono::high_resolution_clock::now();
//   avelableTIME += endS1 - startS1;
//
//   return avilableTransitions;
// }
//
std::vector<Transition> getAvilableDetransitions(const std::unordered_map<std::string, int>& marking) {
   std::vector<Transition> availableDetransitions;
   availableDetransitions.reserve(petri.Transitions.size());

   for (const auto& transition : petri.Transitions) {
     bool canUndo = true;

     // Check if this transition can be undone
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

/*
RCPSPState::RCPSPState() {
   auto startS1 = std::chrono::high_resolution_clock::now();
   //status=true;
direction=true;
   startedActivitiys[0]=0;
  for (int i=0;i<petri.places.size();i++) {
    if (petri.places[i].arcs_out.size()==0){finalstatename=petri.places[i].name;}
    if (petri.places[i].arcs_in.size()==0){initialstatename=petri.places[i].name;}
  }
   for (int i=1;i<petri.Transitions.size();i++) {
     unstartedTransitions.push_back(i+1);
   }
  for (int i=0;i<petri.places.size();i++) {
    if (petri.places[i].name==initialstatename) {
      marking[petri.places[i].name]= 1;
    }
    else {
      marking[petri.places[i].name]= petri.places[i].state[0][0];
    }
  }
  // for (int i=0;i<petri.Transitions.size();i++) {
  //   unstartedTransitions[petri.places[i].name]= 1;
  // }
    auto endS1 = std::chrono::high_resolution_clock::now();

   generateTIME += endS1-startS1;
   avilableTransition = getAvilableTransitions(marking);

  g=0;
  name=0;




}
*/

/*
RCPSPState::RCPSPState(RCPSPState predecesor, Transition active,bool status,int location,uint64_t &count) {
   auto startS4 = std::chrono::high_resolution_clock::now();

   // Only copy what's needed, use references where possible
   direction = predecesor.direction;
   name = count;
   nodestatus = status;

   // Copy containers that will be modified (potentially more efficient than copy+modify)

   unstartedTransitions = predecesor.unstartedTransitions;
   startedActivitiys = predecesor.startedActivitiys;
   finishedActivitiys = predecesor.finishedActivitiys;
   marking = predecesor.marking;
   activeTransitions = predecesor.activeTransitions;
   avilableTransition = predecesor.avilableTransition;
   g=predecesor.g;

  //finalstatename=predecesor.finalstatename;
   //predecesorname=predecesor.name;
  //avilableTransition.erase(avilableTransition.begin()+location);
  //nodestatus=status;
   auto endS1 = std::chrono::high_resolution_clock::now();

   generateTIME += endS1-startS4;
if (direction){
  if (status) {
    h=predecesor.h;

    for (const auto& arc : active.arcs_in) {
       marking[arc.first]-=arc.second;
     }
    //std::cout<<"activate:"<<active.name<<std::endl;
    activeTransitions.push_back(active);

    //if (active.duration==0){status=false;}
    startedActivitiys[active.name]=g;


  }
 else {
   g+=active.duration;

   finishedActivitiys[active.name]=g;
   auto startS2 = std::chrono::high_resolution_clock::now();

   //cureTime+=active.duration;
   unstartedTransitions.erase(
       std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), active.name),
       unstartedTransitions.end());
   //probebly can improve

   for (int i = activeTransitions.size() - 1; i >= 0; --i) {
     activeTransitions[i].duration -= active.duration;
    //  if (activeTransitions[i].duration <= 0) {
    //
    //    finishedActivitiys[active.name]=g-activeTransitions[i].duration;
    //
    //    unstartedTransitions.erase(
    // std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), activeTransitions[i].name),
    // unstartedTransitions.end());
    //
    //    activeTransitions[i].duration=0;
    //
    //    for (const auto& arc : activeTransitions[i].arcs_out) {
    //      marking[arc.first]+=arc.second;
    //
    //    }
    //    activeTransitions.erase(activeTransitions.begin() + i);
    //
    //  }
     //if (activeTransitions[i].duration<0){std::cout<<"!!!!!!!!!!!";}

     if (activeTransitions[i].name == active.name) {
        for (const auto& arc : activeTransitions[i].arcs_out) {
          marking[arc.first]+=arc.second;

        }
        activeTransitions.erase(activeTransitions.begin() + i);
      }

   }
 }

  }
else {
  if (status) {
    h=predecesor.h;

    for (const auto& arc : active.arcs_out) {
      marking[arc.first]-=arc.second;
    }
    //std::cout<<"activate:"<<active.name<<std::endl;
    activeTransitions.push_back(active);

    //if (active.duration==0){status=false;}
    startedActivitiys[active.name]=g;

  }

  else {
    g+=active.duration;

    finishedActivitiys[active.name]=g;
    auto startS2 = std::chrono::high_resolution_clock::now();

    //cureTime+=active.duration;
    unstartedTransitions.erase(
        std::remove(unstartedTransitions.begin(), unstartedTransitions.end(), active.name),
        unstartedTransitions.end());
    //probebly can improve
    for (int i = activeTransitions.size() - 1; i >= 0; --i) {
      activeTransitions[i].duration -= active.duration;
      //if (activeTransitions[i].duration<0){std::cout<<"!!!!!!!!!!!";}
      if (activeTransitions[i].name == active.name) {
        for (const auto& arc : activeTransitions[i].arcs_in) {
          marking[arc.first]+=arc.second;
        }
        activeTransitions.erase(activeTransitions.begin() + i);
      }
    }

  }



  }


   if (direction==true){


     avilableTransition=getAvilableTransitions(marking);

   }
   else {
     auto endS1 = std::chrono::high_resolution_clock::now();

     generateTIME += endS1-startS4;

     avilableTransition=getAvilableDetransitions(marking);

   }

   // if (predecesor.name==20974) {
   //   int qwe;
   //   qwe++;
   // }

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
      h=getForwardHcost(unstartedTransitions,activeTransitionIndices);

    }
  }

  // You'll need to modify these functions to return indices instead of Transitions
  if (direction) {
    avilableTransitionIndices = getAvilableTransitionIndices(marking);

  }
  else {

    avilableTransitionIndices = getAvilableDetransitionIndices(marking);
  }
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
