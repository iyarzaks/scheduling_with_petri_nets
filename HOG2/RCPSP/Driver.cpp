//
// Created by idol on 29/12/2024.
//
// Your First C++ Program

#include <iostream>
 #include "RCPSPState.cpp"
#include "../generic/TemplateAStar.h"
#include "RCPSP.h"
//****importent i changed GLUtil.h with recVec == operator abit****//
 //PetriExample petri;
 //RCPSP_example RCPSP1;
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

std::atomic<bool> stop_printing1(false); // Flag to stop the printing thread

void printNetworkSize1() {
    while (!stop_printing) {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Wait for a second
        std::cout << "Current network size: " << Nsize << std::endl;
    }
}

 int main() {
     getPetri(petri);
     getRCPSP(RCPSPex);

    RCPSPState first;
    RCPSPState last=first;
    for ( auto& pair : last.marking) {
        if (pair.second==1){pair.second=0;}
        if (pair.first==last.finalstatename){pair.second=1;}
    }

    int a;
    RCPSP as1;
    TemplateAStar<RCPSPState, int, RCPSP> astar;

    // משתנה לאחסון הנתיב שנמצא
    std::vector<RCPSPState> path;

    // הרצת A* על הדומיין
     for (int i=0;i<first.avilableTransition.size();i++) {
         count++;
         first.sons.push_back(RCPSPState(first,first.avilableTransition[i],1,i,count));
         first.sons.back().name = count;

     }
    std::thread printer(printNetworkSize1);

     astar.GetPath(&as1, first, last, path);

    // הדפסת הנתיב שנמצא
    std::cout << "Path found!" << std::endl;
    for (const auto& state : path) {
        std::cout <<"time:"<<state.g<< "\n";
        std::cout << "\nsons: ";
        for (int i=0;i<state.sons.size();i++) {
            std::cout<<state.sons[i].name<<" ";
        }
        std::cout << "\nActions in Progress: ";

        for (int i=0;i<state.activeTransitions.size();i++) {
            std::cout<<state.activeTransitions[i].name<<" ";
        }
        std::cout << "\nActions Not Started: ";
        for (const auto& action: state.unstartedTransitions) {
            std::cout << action.first << ":"<<action.second<<" ";
        }
        std::cout <<state.name<< "\n";
    }


    return 0;
}
