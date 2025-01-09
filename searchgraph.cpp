//
// Created by idolu on 06/01/2025.
#include <iostream>
#include <vector>
#include "searchgraph.h"

std::vector<Transition> getAvilableTransitions(std::map<std::string, int> marking);
void GetNabor(std::vector<searchNode> &NodeList,int chosenNode,int &count);

PetriExample petri;
RCPSP_example RCPSP;
int main() {
  getPetri(petri);
  getRCPSP(RCPSP);
  searchNode first;
  std::vector<searchNode> network;
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
std::cout<<"finish with total time of:"<<network[j].g<<std::endl;

    return 1;
  }
}
    int g=-1;
for (int j=0;j<network.size();j++) {
  if (network[j].expanded==0) {
    if (network[j].g>g) {
      g=network[j].g;
      i=j;
    }
  }
}
std::cout<<"expanding:"<<g<<std::endl;

  }
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

void GetNabor(std::vector<searchNode> &NodeList,int chosenNode,int &count) {
  if (NodeList[chosenNode].activeTransitions.size()>0) {
    count++;
    int t=0;
    for (int i=0;i<NodeList[chosenNode].activeTransitions.size();i++) {
      if (NodeList[chosenNode].activeTransitions[i].duration<NodeList[chosenNode].activeTransitions[t].duration) {
        t=i;
      }
    }
    Transition active = NodeList[chosenNode].activeTransitions[t];
    NodeList.push_back(searchNode(NodeList[chosenNode],active,0,t,count));
  }

  for (int i=0;i<NodeList[chosenNode].avilableTransition.size();i++) {
    count++;
    NodeList.push_back(searchNode(NodeList[chosenNode],NodeList[chosenNode].avilableTransition[i],1,i,count));
    NodeList[-1].name=count;

 }
}

searchNode::searchNode() {
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

  avilableTransition=getAvilableTransitions(marking);
  g=0;
  name=0;
}


searchNode::searchNode(searchNode predecesor, Transition active,bool status,int location,int &count) {
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
//h=get(h)
}

int searchNode::GetG() {
  return g;
}

int searchNode::checkEnd() {
  if (marking[finalstatename]==1) {std::cout<<"end"<<std::endl;
  return 1;
  }
  else
    return 0;
}

