//
// Created by idol on 29/12/2024.
//
// Your First C++ Program

#include <iostream>
 #include "RCPSPState.cpp"
#include "../../HOG2/generic/TemplateAStar.h"
#include "RCPSP.h"
//****importent i changed GLUtil.h with recVec == operator abit****//
 //PetriExample petri;
 //RCPSP_example RCPSP1;
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <iostream>
#include <fstream>
#include <vector>

std::atomic<bool> stop_printing1(false); // Flag to stop the printing thread

void printNetworkSize1() {
    while (!stop_printing1) {
        std::this_thread::sleep_for(std::chrono::seconds(60*5)); // Wait for a second
    }
}
int solveRCPSP();
int solveRCPSP(int group,int exam,const std::string& filename) {
    getPetri(petri,group,exam);
    getRCPSP(RCPSPex,group,exam);
    RCPSPState first;
    RCPSPState last=first;
    last.h=0;
    for ( auto& pair : last.marking) {
        if (pair.second==1){pair.second=0;}
        if (pair.first==finalstatename){pair.second=1;}
    }
    int a=0;
    RCPSP as1;
    TemplateAStar<RCPSPState, int, RCPSP> astar;

    // משתנה לאחסון הנתיב שנמצא
    std::vector<RCPSPState> path;

    // הרצת A* על הדומיין
    // for (int i=0;i<first.avilableTransition.size();i++) {
    //     count++;
    //     first.sons.push_back(RCPSPState(first,first.avilableTransition[i],1,i,count));
    //     first.sons.back().name = count;
    //
    // }
    std::thread printer(printNetworkSize1);
    bool finished=false;

    auto start = std::chrono::high_resolution_clock::now();
    astar.GetPath(&as1, first, last, path);
    auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;    // הדפסת הנתיב שנמצא

    if (path.size()>0) {
        std::cout << "Path found!" << std::endl;
        for (const auto& state : path) {


            std::cout << "Actions in Progress: ";
            for (int i = 0; i < state.activeTransitions.size(); i++) {
                std::cout << state.activeTransitions[i].name << " ";
            }
            std::cout << std::endl;

            std::cout << "Available Transitions: ";
            for (int i = 0; i < state.avilableTransition.size(); i++) {
                std::cout << state.avilableTransition[i].name << " ";
            }
            std::cout << std::endl;

            std::cout <<"name:" <<state.name << std::endl;
            std::cout << "g:" << state.g<< std::endl;
            std::cout << "h:" << state.h<< std::endl<< std::endl;
        }
        std::cout <<"end"<<std::endl;
    }
    else{
        std::cout<<"path not found"<< "\n";
        std::cout<<count<< "\n";
    }
    std::cout<<astar.GetNodesExpanded()<<std::endl;
    std::cout<<astar.GetNodesTouched()<<std::endl;
    //std::cout<<astar.GetUniqueNodesExpanded()<<std::endl;
    //std::cout<<path.size()<<std::endl;
    //astar.openClosedList;
    //astar.openClosedList;
    std::ofstream file(filename, std::ios::app);
    std::string status;
    if (finished) {
        status="True";
    }
    else {
        status="False";
    }
    std::vector<std::string> data = {
        std::to_string(group),
        std::to_string(exam),
       std::to_string(elapsed.count()),
        status,  // Ensure status is a single word (e.g., "True" or "False")
        std::to_string(astar.GetNodesExpanded()),
        std::to_string(astar.GetNodesTouched())
    };

    // Write data in a single row
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i < data.size() - 1) file << ",";  // Add comma except at the end
    }
    std::cout << "Data written to " << filename << " successfully!" << std::endl;
    file.close();

    //file << std::endl;  // Only one newline at the end of the row
    exit(0) ;


}
 int main() {
    std::string filename = "output.csv";

    // Open file stream
    std::ofstream file(filename);

    // Check if file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Write header
    file << "group,exam,time,finished,expand number,generated number" << std::endl;

    if (1) {
        solveRCPSP(11,10,filename);

    }
    else {
        for (int i=1;i<5;i++) {
            for (int j=1;j<10;j++) {
                solveRCPSP(i,j,filename);
            }
        }
    }


    return 0;


}
