//
// Created by idolu on 06/01/2025.
#include <iostream>
#include <vector>
#include "searchgraph.h"

std::vector<int> getAvilableTransitions(PetriExample& petri,std::map<std::string, int> marking);
int main() {
  int totalDuration=0;
  int counter=1;
  PetriExample petri;
  getPetri(petri);
  RCPSP_example RCPSP;
  getRCPSP(RCPSP);

  std::string finalstatename;
  std::string initialstatename;
  for (int i=0;i<petri.places.size();i++) {
    if (petri.places[i].arcs_out.size()==0){finalstatename=petri.places[i].name;}
    if (petri.places[i].arcs_in.size()==0){initialstatename=petri.places[i].name;}
  }

  std::map<std::string, int> marking;
  for (int i=0;i<petri.places.size();i++) {
    if (petri.places[i].name==initialstatename) {
      marking[petri.places[i].name]= 1;
    }
    else {
      marking[petri.places[i].name]= petri.places[i].state[0][0];
    }
  }

  std::cout<<"initial state"<<std::endl;
  std::cout<<"current marking"<<std::endl;
  for (const auto& mark : marking) {
    if (mark.second>=1){std::cout<<mark.first<<":"<<mark.second<<" ";}
  }
  std::cout<<std::endl;

  std::cout<<"initial avilable transitions"<<std::endl;
  std::vector<int> avilableTransitions=getAvilableTransitions(petri,marking);
  std::vector<Transition> activeTransitions;
  for (int i=0;i<avilableTransitions.size();i++) {
    std::cout<<petri.Transitions[avilableTransitions[i]].name<<" ";
  }
  std::cout<<std::endl;

  avilableTransitions=getAvilableTransitions(petri,marking);
  std::cout<<"activating transition 1"<<std::endl;

  while (1) {
    std::cout<<std::endl;
    std::cout<<"round number:"<<counter<<std::endl;


std::cout<<"action:";
    if (avilableTransitions.size()==0) {
      int t=0;
      for (int i=1;i<activeTransitions.size();i++) {
        if (activeTransitions[i].duration<activeTransitions[t].duration) {
          t=i;
        }
      }
      Transition active = activeTransitions[t];
      activeTransitions.erase(activeTransitions.begin()+t);
      std::cout<<"ending transition number:"<<active.name<<std::endl;
      for (const auto& arc : active.arcs_out) {
        marking[arc.first]+=arc.second;
      }
      totalDuration+=active.duration;
      for (int i=0;i<activeTransitions.size();i++) {
        activeTransitions[i].duration-=active.duration;
      }
      if (marking[finalstatename]>=1) {
        std::cout<<"win"<<std::endl;
        std::cout<<"totalduration:"<<totalDuration<<std::endl;
        return 0;

      }

    }
    else {
      Transition active = petri.Transitions[avilableTransitions[avilableTransitions.size()-1]];
      std::cout<<"start transition number:"<<active.name<<std::endl;
      activeTransitions.push_back(active);
      for (const auto& arc : active.arcs_in) {
        marking[arc.first]-=arc.second;
      }
    }
    avilableTransitions=getAvilableTransitions(petri,marking);



    std::cout<<"avilable transitions"<<std::endl;
if (avilableTransitions.size()==0) {std::cout<<"None";}

    for (int i=0;i<avilableTransitions.size();i++) {
      std::cout<<petri.Transitions[avilableTransitions[i]].name<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"active transitions"<<std::endl;
    if (activeTransitions.size()==0) {std::cout<<"None";}
    for (int i=0;i<activeTransitions.size();i++) {
      std::cout<<activeTransitions[i].name<<":"<<activeTransitions[i].duration<<" ";
    }
    std::cout<<std::endl;

    std::cout<<"current marking"<<std::endl;
    for (const auto& mark : marking) {
      if (mark.second>=1){std::cout<<mark.first<<":"<<mark.second<<" ";}
    }
    std::cout<<std::endl;

    std::cout<<"current duration:"<<totalDuration<<std::endl;

counter++;
  }

}

//SearchGraph search_graph;
std::vector<int> getAvilableTransitions(PetriExample& petri,std::map<std::string, int> marking) {
  std::vector<int> avilableTransitions;
  for (int i=0;i<petri.Transitions.size();i++) {
    int avilable=0;
    int requirment=0;

    for (const auto& arc : petri.Transitions[i].arcs_in) {
      if (marking[arc.first]>=1){avilable+=arc.second;}
      requirment+=arc.second;
    }
    if (avilable>=requirment) {
      avilableTransitions.insert(avilableTransitions.end(), i);
    }
  }
  return avilableTransitions;
}



SearchGraph::SearchGraph() {

  //std::cout<<petri.places[-1].arcs_out.size();

}
