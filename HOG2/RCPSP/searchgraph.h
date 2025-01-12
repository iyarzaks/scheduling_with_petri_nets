//
// Created by idolu on 06/01/2025.
//
#include "petriclasses.h"
#include "readPetri.cpp"
#ifndef SEARCHGRAPH_H
#define SEARCHGRAPH_H

#endif //SEARCHGRAPH_H
class RCPSPState {
  public:
  RCPSPState(PetriExample petri);
  RCPSPState(RCPSPState predecesor,Transition newTransition,bool status,int location,int &count);
  bool namecomper(Transition a, Transition b);
  std::map<std::string, int> marking;
  std::map<std::string, int> unstartedTransitions;
  std::vector<Transition> avilableTransition;
  std::vector<Transition> activeTransitions;
  std::vector<RCPSPState> sons;
  //std::vector<int> unstartedTransitions;
  bool expanded=0;

  int name;
  std::string finalstatename="_pre_1";
  std::string initialstatename="_pre_1";

  int g;
  int h=0;

  int GetG();
 bool operator==(const RCPSPState &l1, const RCPSPState &l2)
{
   if(l1.expanded!=l2.expanded){
     return false;
   }
    else if(l1.g!=l2.g){
      return false;
    }
    else if(l1.h!=l2.h){
     return false;
    }
    else if(l1.marking!=l2.marking){
      return false;
    }
    else if(l1.unstartedTransitions!=l2.unstartedTransitions){
      return false;
    }
    else if(l1.avilableTransition!=l2.avilableTransition){
      return false;
    }
    else if (l1.activeTransitions.size()==l2.activeTransitions.size()){

      std::sort(l1.activeTransitions.begin(),l1.activeTransitions.end(),namecomper);
      std::sort(l2.activeTransitions.begin(),l2.activeTransitions.end(),namecomper);
      for(int i=0;i<l1.activeTransitions.size();i++) {
        if (l1.activeTransitions[i].name!=l2.activeTransitions[i].name){return false;}
        }
      return true;
    }
   else{return false;}
}

  int checkEnd();

};
bool namecomper(Transition a, Transition b) {
 return a.name<b.name;
}