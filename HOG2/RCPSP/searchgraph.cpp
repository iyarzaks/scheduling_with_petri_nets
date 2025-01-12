//
// Created by idolu on 06/01/2025.
#include <iostream>
#include <vector>
#include "searchgraph.h"

std::vector<Transition> getAvilableTransitions(std::map<std::string, int> marking);
void GetNabor(std::vector<RCPSPState> &NodeList,int chosenNode,int &count);
int ChooseExpansion(std::vector<RCPSPState> network);
PetriExample petri;
RCPSP_example RCPSP;
int main() {
  getPetri(petri);
  getRCPSP(RCPSP);
  RCPSPState first(petri);
  std::vector<RCPSPState> network;
  network.push_back(first);
  //int i;
  int count=0;
  // for (int j=0;j<network.size();j++) {
  //   std::cout<<"Node:"<<network[j].name<<std::endl;
  //   std::cout<<"Avilable Transitions:";
  //   for (int k=0;k<network[j].avilableTransition.size();k++) {std::cout<<network[j].avilableTransition[k].name<<" ";}
  // }
  int i=0;
  while (true) {



    if (network[i].expanded==0) {
      network[i].expanded=1;
      GetNabor(network,i,count);
    }
      for (int j=0;j<network.size();j++) {
        if (network[j].expanded==0) {
          std::cout<<"Node:"<<network[j].name<<std::endl;
          std::cout<<"with g of:"<<network[j].g<<std::endl;
          std::cout<<"activeTransitions:"<<std::endl;
          for (int k=0;k<network[j].activeTransitions.size();k++) {
            std::cout<<network[j].activeTransitions[k].name<<" ";
          }
          std::cout<<std::endl;
          std::cout<<"Avilable Transitions:"<<std::endl;
          for (int k=0;k<network[j].avilableTransition.size();k++) {
            std::cout<<network[j].avilableTransition[k].name<<" ";
          }
          std::cout<<std::endl;
        }
      }
for (int j=0;j<network.size();j++) {
  if (network[j].marking[network[0].finalstatename]==1) {
std::cout<<"finish with total time of:"<<network[j].g;

    return 1;
  }
}
int f=1000;
for (int j=0;j<network.size();j++) {
  if (network[j].expanded==0) {
    if (network[j].g+network[j].h<=f) {
      f=network[j].g+network[j].h;
      i=j;
    }
  }
}
std::cout<<"-----------------"<<std::endl;
std::cout<<"with f of:"<<f<<std::endl;
std::cout<<"expanding:"<<i<<std::endl;

  }
}
int ChooseExpansion(std::vector<RCPSPState> network) {

}

//SearchGraph search_graph;
std::vector<Transition> getAvilableTransitions(std::map<std::string, int> marking) {
  std::vector<Transition> avilableTransitions;
  for (int i=0;i<petri.Transitions.size();i++) {
    int avilable=0;
    int requirment=0;
    int count=0;
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

void GetNabor(std::vector<RCPSPState> &NodeList,int chosenNode,int &count) {
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
    NodeList.back().name = count;

 }
}

RCPSPState::RCPSPState(PetriExample petri) {
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
  for (int i=0;i<petri.Transitions.size();i++) {
    unstartedTransitions[petri.places[i].name]= 1;
  }
  avilableTransition=getAvilableTransitions(marking);
  g=0;
  name=0;
}


RCPSPState::RCPSPState(RCPSPState predecesor, Transition active,bool status,int location,int &count) {
  name=count;
  marking=predecesor.marking;
  activeTransitions=predecesor.activeTransitions;
  avilableTransition=predecesor.avilableTransition;
  finalstatename=predecesor.finalstatename;

  //avilableTransition.erase(avilableTransition.begin()+location);
  g=predecesor.g;
  if (status) {
    for (const auto& arc : active.arcs_in) {
       marking[arc.first]-=arc.second;
     }
    //std::cout<<"activate:"<<active.name<<std::endl;
    activeTransitions.push_back(active);
    unstartedTransitions[active.name]=0;
  }
  else {
    //std::cout<<"ending transition number:"<<active.name<<std::endl;
    for (const auto& arc : active.arcs_out) {
      marking[arc.first]+=arc.second;
    }
    g+=active.duration;
    int temp;
    for (int i=0;i<activeTransitions.size();i++) {
      activeTransitions[i].duration-=active.duration;
      if (activeTransitions[i].name==active.name) {temp=i;}
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

int RCPSPState::GetG() {
  return g;
}

int RCPSPState::checkEnd() {
  if (marking[finalstatename]==1) {std::cout<<"end"<<std::endl;
    return 1;
  }
  else

    return 0;
}

