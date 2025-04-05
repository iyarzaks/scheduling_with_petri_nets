//
// Created by idol on 29/12/2024.
//
// Your First C++ Program

#include <iostream>
 #include "RCPSPState.cpp"
#include "../../HOG2/generic/TemplateAStar.h"
#include "../../HOG2/generic/BAE.h"

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
int solveRCPSP_Bi();
#include <iostream>
#include <fstream>
#include <future>
#include <chrono>
#include <vector>
#include <thread>

namespace fs = std::filesystem;

// Your function signature
int solveRCPSP(int group, int exam, const std::string& filename) {
    std::cout << "started solving: " << group<<":"<<exam << std::endl;

    generateTIME= std::chrono::duration<double>(0);
    avelableTIME= std::chrono::duration<double>(0);
    hashTIME= std::chrono::duration<double>(0);
  //  comperTime= std::chrono::duration<double>(0);
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

    //RCPSP_BiGreedy bs1;


    //BidirectionalGreedyBestFirst<RCPSPState, int, RCPSP_BiGreedy> Bi_RCPSP;
   // Bi_RCPSP.GetPath(&bs1, first, last, fpath,bpath);
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
    auto timeout_point = start + std::chrono::minutes(4
        );

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
int makespan;
    // Output results
    if (finished && !path.empty()) {
        std::cout << "Path found!" << std::endl;
        for (const auto& state : path) {
            std::cout << "g: " << state.g<< std::endl;
            std::cout << "active: ";
            for (int a=0; a<state.activeTransitions.size(); a++) {
                std::cout << " " << state.activeTransitions[a].name;
            }
            std::cout <<  std::endl;
            std::cout << "avilable: ";
            for (int a=0; a<state.avilableTransition.size(); a++) {
                std::cout << " " << state.avilableTransition[a].name;
            }
            std::cout <<  std::endl;
            std::cout <<  std::endl;

        makespan=state.g;
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
         << makespan << ","
         << astar.GetNodesExpanded() << ","
         << astar.GetNodesTouched() << ","<<100*generateTIME.count()/elapsed.count()<< ","<<generateTIME.count()/astar.GetNodesTouched()
             << ","<<100*avelableTIME.count()/elapsed.count()<< ","<<avelableTIME.count()/astar.GetNodesTouched()
                 << ","<<100*hashTIME.count()/elapsed.count()<< ","<<hashTIME.count()/astar.GetNodesTouched()
                 << ","<<100*HTIME.count()/elapsed.count()<< ","<<HTIME.count()/astar.GetNodesTouched()<<
                    // ","<<100*comperTime.count()/elapsed.count()<< ","<<comperTime.count()/astar.GetNodesTouched()<<
             "\n";

    return 0;
}

int solveRCPSP_Bi(int group, int exam, const std::string& filename) {
    std::cout << "started solving: " << group<<":"<<exam << std::endl;

    generateTIME= std::chrono::duration<double>(0);
    avelableTIME= std::chrono::duration<double>(0);
    hashTIME= std::chrono::duration<double>(0);
  //  comperTime= std::chrono::duration<double>(0);
    //secssesorTIME= std::chrono::duration<double>(0);
    count=0;

    getPetri(petri, group, exam);
    getRCPSP(RCPSPex, group, exam);

    RCPSPState first;
    first.direction=true;
    count=2;
    RCPSPState last = first;
    last.direction=false;
    last.h = 0;
last.name=1;
    for (auto& pair : last.marking) {
        if (pair.first=="R1"){continue;}
        if (pair.first=="R2"){continue;}
        if (pair.first=="R3"){continue;}
        if (pair.first=="R4"){continue;}
        if (pair.second == 1) { pair.second = 0; }
        if (pair.first == finalstatename) { pair.second = 1; }
    }
    last.avilableTransition=getAvilableDetransitions(last.marking);


    std::vector<RCPSPState> path;
    ForwardRCPSPHeuristic H_F;
    BackwardRCPSPHeuristic H_B;
    RCPSP_BiGreedy bs1;


    BAE<RCPSPState, int, RCPSP_BiGreedy> Bi_RCPSP;


    Bi_RCPSP.GetPath(&bs1, first, last,&H_F,&H_B ,path);
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
    int makespan;
    // Output results
    if (finished && !path.empty()) {
        std::cout << "Path found!" << std::endl;
        for (const auto& state : path) {
            std::cout << ", g: " << state.g << ", h: " << state.h << std::endl;
        makespan=state.g;
        }
    } else {
        std::cout << "Path not found or timeout occurred.\n";
    }

    std::cout << "Nodes Expanded: " << Bi_RCPSP.GetNodesExpanded() << std::endl;

    // Save to file
    std::ofstream file(filename, std::ios::app);
    file << group << "," << exam << "," << elapsed.count() << ","
         << (finished ? "True" : "False") << ","
         << makespan << ","
         << Bi_RCPSP.GetNodesExpanded() << ","
          << ","<<100*generateTIME.count()/elapsed.count()<< ","<<generateTIME.count()
             << ","<<100*avelableTIME.count()/elapsed.count()<< ","<<avelableTIME.count()
                 << ","<<100*hashTIME.count()/elapsed.count()<< ","<<hashTIME.count()<<
                    // ","<<100*comperTime.count()/elapsed.count()<< ","<<comperTime.count()/astar.GetNodesTouched()<<
             "\n";

    return 0;
}
std::string getNextFilename(const std::string& folder, const std::string& baseName, const std::string& extension) {
    // Ensure folder exists
    if (!fs::exists(folder)) {
        fs::create_directories(folder);
    }

    int count = 1;
    std::string newFilename;

    do {
        newFilename = folder + "/" + baseName + std::to_string(count) + extension;
        count++;
    } while (fs::exists(newFilename)); // Ensure unique filename

    return newFilename;
}

 int main() {
    std::string folder = "results";
    std::string baseName = "output_";
    std::string extension = ".csv";

    std::string filename = getNextFilename(folder, baseName, extension);

    // Create and write to file
    std::ofstream file(filename);
    // Open file stream

    // Check if file is open
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Write header
    file << "group,exam,time,finished,makespan,expand number,generated number,generatedTime%,generatedTime(ave),avilableTime%,avilableTime(ave),hashTime%,hashTime(ave)<<HcostTime%,HcostTime(ave)" << std::endl;

    if (1) {
        //solveRCPSP_Bi(16,9,filename);
        //
       // solveRCPSP(8,9,filename);
       // solveRCPSP(47,1,filename);
       solveRCPSP(38,7,filename);
     //  solveRCPSP(46,1,filename);
   //   solveRCPSP(43,3,filename);
        //solveRCPSP(16,9,filename);
       // solveRCPSP(44,8,filename);
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
