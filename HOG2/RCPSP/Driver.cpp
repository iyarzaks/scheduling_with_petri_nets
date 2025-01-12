//
// Created by idol on 29/12/2024.
//
// Your First C++ Program

#include <iostream>
#include "TemplateAStar.h"

#include "RCPSP.h"

PetriExample petri;
RCPSP_example RCPSP1;
int main() {
    getPetri(petri);
    getRCPSP(RCPSP1);
    RCPSPState first(PetriExample petri);
    int a;
    RCPSP s;
    //SearchEnvironment<RCPSPState,int> RCPSP;
    TemplateAStar<RCPSPState, int, RCPSP> astar;

    // משתנה לאחסון הנתיב שנמצא
    std::vector<RCPSPState> path;

    // הרצת A* על הדומיין
    astar.GetPath(&s, first, first, path);

    // הדפסת הנתיב שנמצא
    std::cout << "Path found:" << std::endl;
    // for (const auto& state : path) {
    //     std::cout << "State: ";
    //     std::cout << "Neighbors: ";
    //     for (int sons : state.sons) {
    //         std::cout << sons << " ";
    //     }
    //     std::cout << "\nActions in Progress: ";
    //     for (int action : state.activeTransitions) {
    //         std::cout << action << " ";
    //     }
    //     std::cout << "\nActions Not Started: ";
    //     for (int action : state.unstartedTransitions) {
    //         std::cout << action << " ";
    //     }
    //     std::cout << "\n";
    // }

    return 0;
}