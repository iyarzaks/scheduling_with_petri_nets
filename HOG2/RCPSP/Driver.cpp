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
#include <windows.h>

#include <fstream>
#include <vector>





std::atomic<bool> stop_printing1(false); // Flag to stop the printing thread

// void printNetworkSize1() {
//     while (!stop_printing1) {
//         std::this_thread::sleep_for(std::chrono::seconds(60*5)); // Wait for a second
//     }
// }
int solveRCPSP();
#include <iostream>
#include <fstream>
#include <future>
#include <chrono>
#include <vector>
#include <thread>

// Your function signature
int solveRCPSP(int group, int exam, const std::string& filename) {
    std::cout << "started solving: " << group<<":"<<exam << std::endl;

    generateTIME= std::chrono::duration<double>(0);
    avelableTIME= std::chrono::duration<double>(0);
    hashTIME= std::chrono::duration<double>(0);
    //secssesorTIME= std::chrono::duration<double>(0);
    count=0;

    getPetri(petri, group, exam);
    getRCPSP(RCPSPex, group, exam);

    RCPSPState first;
    RCPSPState last = first;
    last.h = 0;

    for (auto& pair : last.marking) {
        if (pair.second == 1) { pair.second = 0; }
        if (pair.first == finalstatename) { pair.second = 1; }
    }

    RCPSP as1;
    TemplateAStar<RCPSPState, int, RCPSP> astar;
    std::vector<RCPSPState> path;

    bool finished = false;
    bool timeout_occurred = false;
    std::chrono::duration<double> elapsed;

    // Create a flag for thread completion
    std::atomic<bool> thread_completed(false);

    auto start = std::chrono::high_resolution_clock::now();

    // Create Windows thread handle
    HANDLE win_thread_handle = NULL;

    // Run A* in a separate thread
    std::thread astar_thread([&]() {
        // Get thread handle for potential termination
        DuplicateHandle(
            GetCurrentProcess(),
            GetCurrentThread(),
            GetCurrentProcess(),
            &win_thread_handle,
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS
        );

        // Run the A* algorithm
        astar.GetPath(&as1, first, last, path);

        // Set completion flag when done
        thread_completed = true;
    });

    // Detach the thread so we don't need to join it
    astar_thread.detach();

    // Create a time point for when the timeout should occur
    auto timeout_point = start + std::chrono::minutes(1);

    // Check periodically if the thread has completed or we've reached timeout
    while (!thread_completed && std::chrono::high_resolution_clock::now() < timeout_point) {
        // Short sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Record end time and calculate elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    // Check if timeout occurred
    timeout_occurred = !thread_completed;

    // If timeout occurred, terminate the thread
    if (timeout_occurred && win_thread_handle != NULL) {
        TerminateThread(win_thread_handle, 1);
        std::cout << "Timeout! A* took too long.\n";
        finished = false;
    } else {
        // Thread completed successfully
        finished = true;
    }

    // Close the handle if it exists
    if (win_thread_handle != NULL) {
        CloseHandle(win_thread_handle);
    }

    // Output results
    if (finished && !path.empty()) {
        std::cout << "Path found!" << std::endl;
        for (const auto& state : path) {
            std::cout << "State name: " << state.name << ", g: " << state.g << ", h: " << state.h << std::endl;
        }
    } else {
        std::cout << "Path not found or timeout occurred.\n";
    }

    std::cout << "Nodes Expanded: " << astar.GetNodesExpanded() << std::endl;
    std::cout << "Nodes Touched: " << astar.GetNodesTouched() << std::endl;

    // Save to file
    std::ofstream file(filename, std::ios::app);
    file << group << "," << exam << "," << elapsed.count() << ","
         << (finished ? "True" : "False") << ","
         << astar.GetNodesExpanded() << ","
         << astar.GetNodesTouched() << ","<<100*generateTIME.count()/elapsed.count()<< ","<<generateTIME.count()/astar.GetNodesTouched()
             << ","<<100*avelableTIME.count()/elapsed.count()<< ","<<avelableTIME.count()/astar.GetNodesTouched()
                 << ","<<100*hashTIME.count()/elapsed.count()<< ","<<hashTIME.count()/astar.GetNodesTouched()<<
             "\n";

    return 0;
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
    file << "group,exam,time,finished,expand number,generated number,generatedTime%,generatedTime(ave),avilableTime%,avilableTime(ave),hashTime%,hashTime(ave)" << std::endl;

    if (1) {
        //solveRCPSP(36,4,filename);
        //solveRCPSP(46,1,filename);
        solveRCPSP(43,3,filename);
        //solveRCPSP(16,9,filename);
        //solveRCPSP(44,8,filename);
        //solveRCPSP(38,7,filename);
        //solveRCPSP(11,4,filename);

    }
    else {
        for (int i=10;i<16;i++) {
            for (int j=1;j<11;j++) {
                solveRCPSP(i,j,filename);
            }
        }
    }


    return 0;


}
