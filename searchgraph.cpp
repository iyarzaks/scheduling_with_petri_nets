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
  for (int i=0;i<avilableTransitions.size();i++) {
    std::cout<<petri.Transitions[avilableTransitions[i]].name<<" ";
  }
  std::cout<<std::endl;

  std::cout<<"activating transition 1"<<std::endl;
  while (1) {
    std::cout<<"round number:"<<counter<<std::endl;
    std::cout<<"activating transition number:"<<petri.Transitions[avilableTransitions[0]].name<<std::endl;


    //activate tarnsiotion only avelabe is 1
    Transition active = petri.Transitions[avilableTransitions[0]];
    totalDuration+=active.duration;
    for (const auto& arc : active.arcs_out) {
      marking[arc.first]+=arc.second;
    }
    for (const auto& arc : active.arcs_in) {
      marking[arc.first]-=arc.second;
    }

    std::cout<<"avilable transitions"<<std::endl;
    avilableTransitions=getAvilableTransitions(petri,marking);

    for (int i=0;i<avilableTransitions.size();i++) {
      std::cout<<petri.Transitions[avilableTransitions[i]].name<<" ";
    }
    std::cout<<std::endl;
    std::cout<<"current marking"<<std::endl;
    for (const auto& mark : marking) {
      if (mark.second>=1){std::cout<<mark.first<<":"<<mark.second<<" ";}
    }
    std::cout<<std::endl;
    if (marking[finalstatename]>=1) {
      std::cout<<"win"<<std::endl;
      std::cout<<"totalduration:"<<totalDuration<<std::endl;
      return 0;
    }
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
