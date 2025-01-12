//
// Created by idolu on 06/01/2025.
//
#include "petriclasses.h"
#include "readPetri.cpp"
#ifndef SEARCHGRAPH_H
#define SEARCHGRAPH_H

#endif //SEARCHGRAPH_H
class searchNode {
  public:
  searchNode();
  searchNode(searchNode predecesor,Transition newTransition,bool status,int location,int &count);

  std::map<std::string, int> marking;
  std::map<std::string, int> unstartedTransitions;
  std::vector<Transition> avilableTransition;
  std::vector<Transition> activeTransitions;
  //std::vector<int> unstartedTransitions;
  bool expanded=0;

  int name;
  std::string finalstatename;
  std::string initialstatename;

  int g;
  int h=0;

  int GetG();

  int checkEnd();

};
class SearchGraph{
  public:
  ~SearchGraph();
  SearchGraph();



};
