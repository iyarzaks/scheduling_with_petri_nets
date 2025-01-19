//
// Created by idolu on 06/01/2025.

#include <iostream>
#include <vector>
#include "RCPSPState.h"
#include <thread>
#include <chrono>
#include <atomic>

// std::atomic<bool> stop_printing(false); // Flag to stop the printing thread
//
// void printNetworkSize(const std::vector<RCPSPState>& network) {
//   while (!stop_printing) {
//     std::this_thread::sleep_for(std::chrono::seconds(1)); // Wait for a second
//     std::cout << "Current network size: " << network.size() << std::endl;
//   }
// }
std::vector<Transition> getAvilableTransitions(std::map<std::string, int> marking);
void GetNabor(std::vector<RCPSPState> &NodeList,int chosenNode,int &count);
//int ChooseExpansion(std::vector<RCPSPState> network);
 PetriExample petri;
 RCPSP_example RCPSPex;
 int main2() {
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
std::vector<Transition> getAvilableTransitions(std::map<std::string, int> marking) {
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
  return avilableTransitions;
}

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

RCPSPState::RCPSPState() {
  for (int i=0;i<petri.places.size();i++) {
    if (petri.places[i].arcs_out.size()==0){finalstatename=petri.places[i].name;}
    if (petri.places[i].arcs_in.size()==0){initialstatename=petri.places[i].name;}
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
  avilableTransition=getAvilableTransitions(marking);
  g=0;
  name=0;

}


RCPSPState::RCPSPState(RCPSPState predecesor, Transition active,bool status,int location,uint64_t &count) {
  name=count;
  marking=predecesor.marking;
  activeTransitions=predecesor.activeTransitions;
  avilableTransition=predecesor.avilableTransition;
  //finalstatename=predecesor.finalstatename;

  //avilableTransition.erase(avilableTransition.begin()+location);
  g=predecesor.g;
  if (status) {
    for (const auto& arc : active.arcs_in) {
       marking[arc.first]-=arc.second;
     }
    //std::cout<<"activate:"<<active.name<<std::endl;
    activeTransitions.push_back(active);
    //unstartedTransitions[active.name]=0;
  }
  else {
    //std::cout<<"ending transition number:"<<active.name<<std::endl;
    for (const auto& arc : active.arcs_out) {
      marking[arc.first]+=arc.second;
    }
    g+=active.duration;
    int temp;

    //probebly can improve
    for (int i=0;i<activeTransitions.size();i++) {
      activeTransitions[i].duration-=active.duration;
      if (activeTransitions[i].name==active.name) {temp=i;break;}
    }

    activeTransitions.erase(activeTransitions.begin()+temp);
  }
  avilableTransition=getAvilableTransitions(marking);
  // for (int i=0;i<petri.Transitions.size();i++) {
  //   if (unstartedTransitions[petri.Transitions[i].name]==1) {
  //
  //   }
  // }

//h=get(h)
}

// int RCPSPState::GetG() {
//   return g;
// }

// int RCPSPState::checkEnd() {
//   if (marking[finalstatename]==1) {std::cout<<"end"<<std::endl;
//     return 1;
//   }
//   else
//
//     return 0;
// }

// bool operator==(const RCPSPState &l1, const RCPSPState &l2) {
//    if (l1.expanded != l2.expanded) {
//      return false;
//    }
//    if (l1.g != l2.g) {
//      return false;
//    }
//    if (l1.h != l2.h) {
//      return false;
//    }
//    // if (l1.marking != l2.marking) {
//    //     return false;
//    // }
//    // if (l1.unstartedTransitions != l2.unstartedTransitions) {
//    //     return false;
//    // }
//    if (l1.avilableTransition != l2.avilableTransition) {
//      return false;
//    }
//    // if (l1.activeTransitions.size() == l2.activeTransitions.size()) {
//    //     // Make local copies of the transitions for sorting
//    //     std::vector<Transition> sortedL1Transitions = l1.activeTransitions;
//    //     std::vector<Transition> sortedL2Transitions = l2.activeTransitions;
//    //
//    //     // Sort the copies
//    //     std::sort(sortedL1Transitions.begin(), sortedL1Transitions.end(), namecomper);
//    //     std::sort(sortedL2Transitions.begin(), sortedL2Transitions.end(), namecomper);
//    //
//    //     // Compare the sorted transitions
//    //     for (size_t i = 0; i < sortedL1Transitions.size(); i++) {
//    //         if (sortedL1Transitions[i].name != sortedL2Transitions[i].name) {
//    //             return false;
//    //         }
//    //     }
//    //     return true;
//    // }
//    return false;
//  }
bool RCPSPState::operator==(const RCPSPState& other) const {
   if (this->expanded != other.expanded) {
     return false;
   }
   if (this->g != other.g) {
     return false;
   }
   if (this->h != other.h) {
     return false;
   }
   if (this->avilableTransition != other.avilableTransition) {
     return false;
   }
   if (this->activeTransitions != other.activeTransitions) {
     return false;
   }
   return true;
 }
