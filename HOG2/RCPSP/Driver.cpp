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
// #include <windows.h>

#include <fstream>
#include <vector>




void runBenchmark();
void runSolvedProblems();
std::atomic<bool> cancel_requested(false);

std::atomic<bool> stop_printing1(false); // Flag to stop the printing thread

// void printNetworkSize1() {
//     while (!stop_printing1) {
//         std::this_thread::sleep_for(std::chrono::seconds(60*5)); // Wait for a second
//     }
// }
int solveRCPSP();
int solveRCPSP_TT();
int solveRCPSP_Bi();
#include <iostream>
#include <fstream>
#include <future>
#include <chrono>
#include <vector>
#include <thread>
#include <thread>
#include <atomic>
namespace fs = std::filesystem;

// Your function signature
/*
int solveRCPSP_old(int group, int exam, const std::string& filename) {
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
    auto timeout_point = start + std::chrono::minutes(1
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
                 << ","<<100*HTIME.count()/elapsed.count()<< ","<<HTIME.count()/count<<
                   //  ","<<100*comperTime.count()/elapsed.count()<< ","<<comperTime.count()/astar.GetNodesTouched()<<
             "\n";

    return 0;
}
*/\
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <future>
#include <atomic>
#include "RCPSP.h" // assuming these are your own headers

int solveRCPSP(int group, int exam, const std::string& filename,const std::string& problemType="j30") {
    std::cout << "started solving: " << group<<":"<<exam << std::endl;

    //generateTIME= std::chrono::duration<double>(0);
    //avelableTIME= std::chrono::duration<double>(0);
    //hashTIME= std::chrono::duration<double>(0);
    //  comperTime= std::chrono::duration<double>(0);
    //secssesorTIME= std::chrono::duration<double>(0);
    count=0;
    getPetri(petri, group, exam,problemType);
    getRCPSP(RCPSPex, group, exam,problemType);
    // getPetri(petri, group, exam);
    // getRCPSP(RCPSPex, group, exam);
    RCPSPex.computeAndStoreDeepDependencies();
    RCPSPState first;
    RCPSPState last = first;
    last.h = 0;

    for (auto& pair : last.marking) {
        if (pair.second == 1) { pair.second = 0; }
        if (pair.first == finalstatename) { pair.second = 1; }
    }
    for (const auto& pair : last.marking) {
        const std::string& name = pair.first;
        int value = pair.second;

        // if (name == "R1" || name == "R2" || name == "R3" || name == "R4") {
        //     RCPSPex.addResource(name, value);
        // }
    }
    RCPSP as1;
    TemplateAStar<RCPSPState, int, RCPSP> astar;
    std::vector<RCPSPState> path;


    bool finished = false;
    bool timeout_occurred = false;
    std::chrono::duration<double> elapsed;

    auto start = std::chrono::high_resolution_clock::now();
    astar.GetPath(&as1, first, last, path);
    // Use std::async to run A* in a separate thread with future
    // auto future = std::async(std::launch::async, [&]() {
    //
    // });
    //
    // // Wait for up to 5 minutes
    // if (future.wait_for(std::chrono::minutes(10)) == std::future_status::timeout) {
    //     timeout_occurred = true;
    //     std::cout << "Timeout! A* took too long.\n";
    //     finished = false;
    // } else {
    //     finished = true;
    // }

    auto end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    int makespan = 0;

    if (!path.empty()) {
        std::cout << "Path found!" << std::endl;
        for (const auto& state : path) {
            std::cout << "g: " << state.g << std::endl;

            std::cout << "active: ";
            for (const auto& [transIdx, duration] : state.activeTransitionIndices)
                std::cout << " " << transIdx;
            std::cout << std::endl;

            std::cout << "available: ";
            for (int transIdx : state.avilableTransitionIndices)
                std::cout << " " << transIdx;
            std::cout << std::endl << std::endl;

            makespan = state.g;
        }
    } else {
        std::cout << "Path not found or timeout occurred.\n";
    }

    std::cout << "Nodes Expanded: " << astar.GetNodesExpanded() << std::endl;
    std::cout << "Nodes Touched: " << astar.GetNodesTouched() << std::endl;

    std::ofstream file(filename, std::ios::app);
    file << group << "," << exam << "," << elapsed.count() << ","
         << (!path.empty() ? "True" : "False") << ","
         << makespan << ","
         << astar.GetNodesExpanded() << ","
         << astar.GetNodesTouched()<< ","
         << path.size()
      //   << 100 * generateTIME.count() / elapsed.count() << ","
        // << generateTIME.count() / astar.GetNodesTouched() << ","
         //<< 100 * avelableTIME.count() / elapsed.count() << ","
         //<< avelableTIME.count() / astar.GetNodesTouched() << ","
         //<< 100 * hashTIME.count() / elapsed.count() << ","
         //<< hashTIME.count() / astar.GetNodesTouched() << ","
         //<< 100 * HTIME.count() / elapsed.count() << ","
         //<< HTIME.count() / count
         << "\n";

    return 0;
}
    int solveRCPSP_TT(int group, int exam, const std::string& filename,const std::string& problemType="j30") {
    std::cout << "started solving: " << group<<":"<<exam << std::endl;

    //generateTIME= std::chrono::duration<double>(0);
    //avelableTIME= std::chrono::duration<double>(0);
    //hashTIME= std::chrono::duration<double>(0);
    //  comperTime= std::chrono::duration<double>(0);
    //secssesorTIME= std::chrono::duration<double>(0);
    count=0;
    getPetri(petri, group, exam,problemType);
    getRCPSP(RCPSPex, group, exam,problemType);

    RCPSPState_TT first;
    RCPSPState_TT last = first;
    //last.h = 0;



    RCPSP_TT as1;

    TemplateAStar<RCPSPState_TT, int, RCPSP_TT> astar;
    std::vector<RCPSPState_TT> path;


    std::chrono::duration<double> elapsed;

    auto start = std::chrono::high_resolution_clock::now();
    astar.GetPath(&as1, first, last, path);

    auto end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    int makespan = 0;

    if (!path.empty()) {
        std::cout << "Path found!" << std::endl;

        //for (const auto& state : path) {
        RCPSPState_TT state=path.back();
            //std::cout << "g: " << state.g;

            for (const auto& [actId, startTime] : state.startedActivitiys) {
                std::cout << actId << ":" << startTime << " ";
            }

            std::cout << std::endl;
            makespan = state.g;
        //}

        std::cout << "\nFinal makespan: " << makespan << std::endl;
    }
     else {
        std::cout << "Path not found or timeout occurred.\n";
    }

    std::cout << "Nodes Expanded: " << astar.GetNodesExpanded() << std::endl;
    std::cout << "Nodes Touched: " << astar.GetNodesTouched() << std::endl;

    std::ofstream file(filename, std::ios::app);
    file << group << "," << exam << "," << elapsed.count() << ","
         << (!path.empty() ? "True" : "False") << ","
         << makespan << ","
         << astar.GetNodesExpanded() << ","
         << astar.GetNodesTouched()<< ","
         << path.size()
         << "\n";

    return 0;
}
/*
int solveRCPSP(int group, int exam, const std::string& filename) {


    std::cout << "started solving: " << group << ":" << exam << std::endl;

    generateTIME = std::chrono::duration<double>(0);
    avelableTIME = std::chrono::duration<double>(0);
    hashTIME = std::chrono::duration<double>(0);
    HTIME = std::chrono::duration<double>(0);
    count = 0;

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
    auto timeout_point = start + std::chrono::minutes(5);

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

    int makespan = 0;

    // Output results
    if (finished && !path.empty()) {
        std::cout << "Path found!" << std::endl;
        for (const auto& state : path) {
            std::cout << "g: " << state.g << std::endl;

            // Changed: Display active transitions using indices
            std::cout << "active: ";
            for (const auto& [transIdx, duration] : state.activeTransitionIndices) {
                std::cout << " " << transIdx;
            }
            std::cout << std::endl;

            // Changed: Display available transitions using indices
            std::cout << "available: ";
            for (int transIdx : state.avilableTransitionIndices) {
                std::cout << " " << transIdx;
            }
            std::cout << std::endl << std::endl;

            makespan = state.g;
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
         << astar.GetNodesTouched() << ","
         << 100 * generateTIME.count() / elapsed.count() << ","
         << generateTIME.count() / astar.GetNodesTouched() << ","
         << 100 * avelableTIME.count() / elapsed.count() << ","
         << avelableTIME.count() / astar.GetNodesTouched() << ","
         << 100 * hashTIME.count() / elapsed.count() << ","
         << hashTIME.count() / astar.GetNodesTouched() << ","
         << 100 * HTIME.count() / elapsed.count() << ","
         << HTIME.count() / count << "\n";

    return 0;
}
*/

int solveRCPSP_Bi(int group, int exam, const std::string& filename) {
    std::cout << "started solving: " << group<<":"<<exam << std::endl;

  //  generateTIME= std::chrono::duration<double>(0);
    //avelableTIME= std::chrono::duration<double>(0);
   // hashTIME= std::chrono::duration<double>(0);
  //  comperTime= std::chrono::duration<double>(0);
    //secssesorTIME= std::chrono::duration<double>(0);
    count=0;

    getPetri(petri, group, exam);
    getRCPSP(RCPSPex, group, exam);

    RCPSPState_bi first;
    first.direction=true;
    count=2;
    RCPSPState_bi last = first;
    last.direction=false;
    last.h_b = first.h_f;
    last.h_f = 0;
last.name=1;
    for (auto& pair : last.marking) {
        //set only to to 4 diffrent resources
        if (pair.first=="R1"){continue;}
        if (pair.first=="R2"){continue;}
        if (pair.first=="R3"){continue;}
        if (pair.first=="R4"){continue;}
        if (pair.second == 1) { pair.second = 0; }
        if (pair.first == finalstatename) { pair.second = 1; }
    }
    last.avilableDeTransitionIndices=getAvilableDetransitionIndices(last.marking);
    last.avilableTransitionIndices=getAvilableTransitionIndices(last.marking);
    for (int i=1;i<petri.Transitions.size()+1;i++) {
        last.finishedActivitiys.insert(i);
        last.startedActivitiys.insert(i);
    }
    last.unstartedTransitions.clear();
    std::vector<RCPSPState_bi> path;
    ForwardRCPSPHeuristic H_F;
    BackwardRCPSPHeuristic H_B;
    RCPSP_BiGreedy bs1;


    BAE<RCPSPState_bi, int, RCPSP_BiGreedy> Bi_RCPSP;



    bool finished = false;
    bool timeout_occurred = false;
    std::chrono::duration<double> elapsed;

    // Create a flag for thread completion

    auto start = std::chrono::high_resolution_clock::now();
    Bi_RCPSP.GetPath(&bs1, first, last,&H_F,&H_B ,path);



    // Record end time and calculate elapsed time
    auto end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;


    // Output results
    double max_f = 0;
    double max_b = 0;

    if (!path.empty()) {
        finished=true;
        std::cout << "Path found!" << std::endl;
        for (const auto& state : path) {
            std::cout << "g_f: " << state.g_f << std::endl;
            std::cout << "g_b: " << state.g_b << std::endl;

            std::cout << "active: ";
            for (const auto& [transIdx, duration] : state.activeTransitionIndices)
                std::cout << " " << transIdx;
            std::cout << std::endl;

            std::cout << "available: ";
            for (int transIdx : state.avilableTransitionIndices)
                std::cout << " " << transIdx;
            std::cout << std::endl << std::endl;

            max_f = std::max(max_f,state.g_f);
            max_b = std::max(max_b,state.g_b);
        }
    } else {
        std::cout << "Path not found or timeout occurred.\n";
    }

    std::cout << "Nodes Expanded: " << Bi_RCPSP.GetNodesExpanded() << std::endl;
    std::cout << "Nodes Touched: " << Bi_RCPSP.GetNodesTouched() << std::endl;
    // Save to file
    std::ofstream file(filename, std::ios::app);
    file << group << "," << exam << "," << elapsed.count() << ","
         << (finished ? "True" : "False") << ","
         << max_f+max_b << ","
         << Bi_RCPSP.GetNodesExpanded() << ","
        //  << ","<<100*generateTIME.count()/elapsed.count()<< ","<<generateTIME.count()
         //    << ","<<100*avelableTIME.count()/elapsed.count()<< ","<<avelableTIME.count()
          //       << ","<<100*hashTIME.count()/elapsed.count()<< ","<<hashTIME.count()<<
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
     runBenchmark();
//runSolvedProblems();
    //if (1) {
    //
     //    solveRCPSP_Bi(-1,-1,filename);
    //     //
       // solveRCPSP(8,9,filename);
    //solveRCPSP(-1,-1,filename);
    //     //solveRCPSP(47,1,filename);
    //   // solveRCPSP(38,7,filename);
    //   // solveRCPSP(46,1,filename);
    //   //  solveRCPSP(43,3,filename);
    // //    solveRCPSP(16,9,filename);
    //     //solveRCPSP(44,8,filename);
    //      //solveRCPSP(11,4,filename);
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
     //}
    // else {
    //     for (int i=10;i<16;i++) {
    //         for (int j=1;j<11;j++) {
    //             solveRCPSP(i,j,filename);
    //         }
    //     }
    // }

    return 0;


}

void getinitialHcost(int i, int i1, const std::string & string);
void runSolvedProblems() {
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
        return;
    }

    // Write header
    //file << "group,exam,time,finished,makespan,expand number,generated number,generatedTime%,generatedTime(ave),avilableTime%,avilableTime(ave),hashTime%,hashTime(ave),HcostTime%,HcostTime(ave)" << std::endl;
    file << "group,exam,time,finished,makespan,expand number,generated number,depth" << std::endl;
    // solveRCPSP(1, 7, filename);
    // solveRCPSP(2, 3, filename);
    // solveRCPSP(2, 5, filename);
    // solveRCPSP(2, 7, filename);
    // solveRCPSP(3, 1, filename);
    // solveRCPSP(3, 3, filename);
    // solveRCPSP(3, 4, filename);
    // solveRCPSP(3, 6, filename);
    // solveRCPSP(3, 7, filename);
    // solveRCPSP(3, 8, filename);
    // solveRCPSP(4, 1, filename);
    // solveRCPSP(4, 2, filename);
    // solveRCPSP(4, 4, filename);
    // solveRCPSP(4, 5, filename);
    // solveRCPSP(4, 6, filename);
    // solveRCPSP(4, 7, filename);
    // solveRCPSP(4, 8, filename);
    // solveRCPSP(4, 10, filename);
    // solveRCPSP(6, 6, filename);
    // solveRCPSP(7, 1, filename);
    // solveRCPSP(7, 2, filename);
    // solveRCPSP(7, 3, filename);
    // solveRCPSP(8, 1, filename);
    // solveRCPSP(8, 2, filename);
    // solveRCPSP(8, 3, filename);
    // solveRCPSP(8, 4, filename);
    // solveRCPSP(8, 5, filename);
    // solveRCPSP(8, 6, filename);
    // solveRCPSP(8, 7, filename);
    // solveRCPSP(8, 8, filename);
    // solveRCPSP(8, 9, filename);
    // solveRCPSP(8, 10, filename);
    // solveRCPSP(11, 1, filename);
    // solveRCPSP(11, 3, filename);
    // solveRCPSP(11, 6, filename);
    // solveRCPSP(11, 7, filename);
    // solveRCPSP(11, 10, filename);
    // solveRCPSP(12, 1, filename);
    // solveRCPSP(12, 2, filename);
    // solveRCPSP(12, 3, filename);
    // solveRCPSP(12, 4, filename);
    // solveRCPSP(12, 5, filename);
    // solveRCPSP(12, 6, filename);
    // solveRCPSP(12, 7, filename);
    // solveRCPSP(12, 8, filename);
    // solveRCPSP(12, 9, filename);
    // solveRCPSP(12, 10, filename);
    // solveRCPSP(14, 2, filename);
    // solveRCPSP(14, 6, filename);
    // solveRCPSP(14, 7, filename);
    // solveRCPSP(15, 1, filename);
    // solveRCPSP(15, 3, filename);
    // solveRCPSP(15, 5, filename);
    // solveRCPSP(15, 6, filename);
    // solveRCPSP(15, 8, filename);
    // solveRCPSP(15, 9, filename);
    // solveRCPSP(16, 1, filename);
    // solveRCPSP(16, 3, filename);
    // solveRCPSP(16, 4, filename);
    // solveRCPSP(16, 5, filename);
    // solveRCPSP(16, 6, filename);
    // solveRCPSP(16, 8, filename);
    // solveRCPSP(16, 9, filename);
    // solveRCPSP(17, 4, filename);
    // solveRCPSP(17, 10, filename);
    // solveRCPSP(18, 7, filename);
    // solveRCPSP(19, 2, filename);
    // solveRCPSP(19, 4, filename);
    // solveRCPSP(19, 6, filename);
    // solveRCPSP(19, 7, filename);
    // solveRCPSP(19, 8, filename);
    // solveRCPSP(19, 9, filename);
    // solveRCPSP(20, 1, filename);
    // solveRCPSP(20, 2, filename);
    // solveRCPSP(20, 3, filename);
    // solveRCPSP(20, 4, filename);
    // solveRCPSP(20, 5, filename);
    // solveRCPSP(20, 6, filename);
    // solveRCPSP(20, 7, filename);
    // solveRCPSP(20, 8, filename);
    // solveRCPSP(20, 9, filename);
    // solveRCPSP(20, 10, filename);
    // solveRCPSP(22, 1, filename);
    // solveRCPSP(22, 8, filename);
    // solveRCPSP(23, 1, filename);
    // solveRCPSP(23, 2, filename);
    // solveRCPSP(23, 5, filename);
    // solveRCPSP(23, 6, filename);
    // solveRCPSP(23, 8, filename);
    // solveRCPSP(24, 1, filename);
    // solveRCPSP(24, 2, filename);
    // solveRCPSP(24, 3, filename);
    // solveRCPSP(24, 6, filename);
    // solveRCPSP(24, 8, filename);
    // solveRCPSP(24, 9, filename);
    // solveRCPSP(24, 10, filename);
    // solveRCPSP(26, 2, filename);
    // solveRCPSP(26, 3, filename);
    // solveRCPSP(26, 5, filename);
    // solveRCPSP(26, 6, filename);
    // solveRCPSP(26, 8, filename);
    // solveRCPSP(27, 1, filename);
    // solveRCPSP(27, 2, filename);
    // solveRCPSP(27, 3, filename);
    // solveRCPSP(27, 4, filename);
    // solveRCPSP(27, 5, filename);
    // solveRCPSP(27, 6, filename);
    // solveRCPSP(27, 8, filename);
    // solveRCPSP(27, 9, filename);
    // solveRCPSP(27, 10, filename);
    // solveRCPSP(28, 1, filename);
    // solveRCPSP(28, 2, filename);
    // solveRCPSP(28, 3, filename);
    // solveRCPSP(28, 4, filename);
    // solveRCPSP(28, 5, filename);
    // solveRCPSP(28, 6, filename);
    // solveRCPSP(28, 7, filename);
    // solveRCPSP(28, 8, filename);
    // solveRCPSP(28, 9, filename);
    // solveRCPSP(31, 2, filename);
    // solveRCPSP(31, 3, filename);
    // solveRCPSP(31, 4, filename);
    // solveRCPSP(31, 5, filename);
    // solveRCPSP(31, 7, filename);
    // solveRCPSP(31, 8, filename);
    // solveRCPSP(32, 1, filename);
    // solveRCPSP(32, 2, filename);
    // solveRCPSP(32, 3, filename);
    // solveRCPSP(32, 4, filename);
    // solveRCPSP(32, 5, filename);
    // solveRCPSP(32, 6, filename);
    // solveRCPSP(32, 7, filename);
    // solveRCPSP(32, 8, filename);
    // solveRCPSP(32, 9, filename);
    // solveRCPSP(32, 10, filename);
    // solveRCPSP(33, 1, filename);
    // solveRCPSP(33, 2, filename);
    // solveRCPSP(33, 9, filename);
    // solveRCPSP(34, 2, filename);
    // solveRCPSP(34, 3, filename);
    // solveRCPSP(34, 4, filename);
    // solveRCPSP(34, 10, filename);
    // solveRCPSP(35, 1, filename);
    // solveRCPSP(35, 2, filename);
    // solveRCPSP(35, 3, filename);
    // solveRCPSP(35, 4, filename);
    // solveRCPSP(35, 5, filename);
    // solveRCPSP(35, 6, filename);
    // solveRCPSP(35, 7, filename);
    // solveRCPSP(36, 1, filename);
    // solveRCPSP(36, 2, filename);
    // solveRCPSP(36, 3, filename);
    // solveRCPSP(36, 4, filename);
    // solveRCPSP(36, 5, filename);
    // solveRCPSP(36, 6, filename);
    // solveRCPSP(36, 7, filename);
    // solveRCPSP(36, 8, filename);
    // solveRCPSP(36, 9, filename);
    // solveRCPSP(36, 10, filename);
    // solveRCPSP(38, 2, filename);
    // solveRCPSP(38, 3, filename);
    // solveRCPSP(38, 6, filename);
    // solveRCPSP(39, 1, filename);
    // solveRCPSP(39, 2, filename);
    // solveRCPSP(39, 3, filename);
    // solveRCPSP(39, 5, filename);
    // solveRCPSP(39, 7, filename);
    // solveRCPSP(39, 8, filename);
    // solveRCPSP(39, 9, filename);
    // solveRCPSP(40, 1, filename);
    // solveRCPSP(40, 2, filename);
    // solveRCPSP(40, 3, filename);
    // solveRCPSP(40, 4, filename);
    // solveRCPSP(40, 5, filename);
    // solveRCPSP(40, 6, filename);
    // solveRCPSP(40, 7, filename);
    // solveRCPSP(40, 8, filename);
    // solveRCPSP(40, 9, filename);
    // solveRCPSP(40, 10, filename);
    // solveRCPSP(42, 1, filename);
    // solveRCPSP(42, 5, filename);
    // solveRCPSP(42, 6, filename);
    // solveRCPSP(43, 2, filename);
    // solveRCPSP(43, 4, filename);
    // solveRCPSP(43, 5, filename);
    // solveRCPSP(43, 6, filename);
    // solveRCPSP(43, 9, filename);
    // solveRCPSP(43, 10, filename);
    // solveRCPSP(44, 1, filename);
    // solveRCPSP(44, 2, filename);
    // solveRCPSP(44, 4, filename);
    // solveRCPSP(44, 7, filename);
    // solveRCPSP(44, 8, filename);
    // solveRCPSP(44, 9, filename);
    // solveRCPSP(44, 10, filename);
    // solveRCPSP(46, 3, filename);
    // solveRCPSP(46, 4, filename);
    // solveRCPSP(46, 7, filename);
    // solveRCPSP(46, 9, filename);
    // solveRCPSP(47, 2, filename);
    // solveRCPSP(47, 3, filename);
    // solveRCPSP(47, 5, filename);
    // solveRCPSP(47, 9, filename);
    // solveRCPSP(48, 1, filename);
    // solveRCPSP(48, 2, filename);
    // solveRCPSP(48, 3, filename);
    // solveRCPSP(48, 4, filename);
    // solveRCPSP(48, 5, filename);
    // solveRCPSP(48, 6, filename);
    // solveRCPSP(48, 7, filename);
    // solveRCPSP(48, 8, filename);
    // solveRCPSP(48, 9, filename);
    // solveRCPSP(48, 10, filename);

}
void runBenchmark() {
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
        return;
    }

    // Write header
    //file << "group,exam,time,finished,makespan,expand number,generated number,generatedTime%,generatedTime(ave),avilableTime%,avilableTime(ave),hashTime%,hashTime(ave),HcostTime%,HcostTime(ave)" << std::endl;
    file << "group,exam,time,finished,makespan,expand number,generated number,depth" << std::endl;
    //file << "group,exam,initialHcost" << std::endl;
//

  // getinitialHcost(-1, -1, filename);
  //solveRCPSP(-1,-1,filename);
  //solveRCPSP_Bi(3, 6, filename);
 //solveRCPSP(-1, 1, filename);
// // solveRCPSP_Bi(39, 4, filename);
  // solveRCPSP_TT(16, 9, filename);
     // solveRCPSP_TT(33, 5, filename);
     // solveRCPSP_TT(19, 4, filename);
     // // solveRCPSP_TT(17, 6, filename);
     //  solveRCPSP(16, 9, filename);
     //
     // //solveRCPSP_TT(17, 6, filename);
    //  solveRCPSP(2, 3, filename);
    //  solveRCPSP(2, 7, filename);
    //  solveRCPSP(3, 7, filename);
    //  solveRCPSP(3, 8, filename);
    // solveRCPSP(4, 1, filename);
    // solveRCPSP(4, 2, filename);
    // solveRCPSP(4, 4, filename);
    // solveRCPSP(4, 5, filename);
    // solveRCPSP(4, 6, filename);
    // solveRCPSP(12, 1, filename);
    // solveRCPSP(12, 2, filename);
    // solveRCPSP(12, 3, filename);
    // solveRCPSP(12, 4, filename);
    // solveRCPSP(12, 5, filename);
    // solveRCPSP(18, 2, filename);
    // solveRCPSP(19, 9, filename);
    // solveRCPSP(20, 1, filename);
    // solveRCPSP(20, 2, filename);
    // solveRCPSP(20, 3, filename);
    // solveRCPSP(20, 4, filename);
    // solveRCPSP(28, 7, filename);
    // solveRCPSP(28, 8, filename);
    //
    //   //
    //    solveRCPSP(26, 6, filename);
    //   solveRCPSP(19, 10, filename);

    //

        // for(int j=4;j<11;j++) {
        //     solveRCPSP(25,j,filename);
        //     //   getinitialHcost(i,j,filename);
        // }
    //  for(int j=3;j<11;j++) {
    //      solveRCPSP(21,j,filename);
    //      //   getinitialHcost(i,j,filename);
    // }
    // solveRCPSP(26,6,filename);
    // solveRCPSP_TT(26,6,filename);
    for(int i=1;i<49;i++) {
         for(int j=1;j<11;j++) {
         solveRCPSP(i,j,filename,"j60");
      //   getinitialHcost(i,j,filename);
         }
     }
  //  solveRCPSP_TT(1, 10, filename);
  //  solveRCPSP(8, 9,filename);
    //solveRCPSP_Bi(39, 4, filename);
 // solveRCPSP_TT(6,7,filename);
//     solveRCPSP_TT(1, 9, filename);
// solveRCPSP_TT(1, 10, filename);
// solveRCPSP_TT(2, 3, filename);
// solveRCPSP_TT(2, 8, filename);
// solveRCPSP_TT(2, 10, filename);
// solveRCPSP_TT(6, 3, filename);
// solveRCPSP_TT(6, 7, filename);
// solveRCPSP_TT(6, 10, filename);
// solveRCPSP_TT(7, 4, filename);
// solveRCPSP_TT(7, 5, filename);
// solveRCPSP_TT(7, 7, filename);
// solveRCPSP_TT(10, 2, filename);
// solveRCPSP_TT(10, 4, filename);
// solveRCPSP_TT(10, 5, filename);
// solveRCPSP_TT(10, 7, filename);
// solveRCPSP_TT(10, 10, filename);
// solveRCPSP_TT(11, 1, filename);
// solveRCPSP_TT(11, 2, filename);
// solveRCPSP_TT(11, 5, filename);
// solveRCPSP_TT(11, 7, filename);
// solveRCPSP_TT(11, 10, filename);
// solveRCPSP_TT(12, 1, filename);
// solveRCPSP_TT(14, 1, filename);
// solveRCPSP_TT(14, 3, filename);
// solveRCPSP_TT(14, 5, filename);
// solveRCPSP_TT(14, 6, filename);
// solveRCPSP_TT(14, 10, filename);
// solveRCPSP_TT(17, 2, filename);
// solveRCPSP_TT(17, 7, filename);
// solveRCPSP_TT(17, 8, filename);
// solveRCPSP_TT(17, 9, filename);
// solveRCPSP_TT(19, 9, filename);
// solveRCPSP_TT(22, 1, filename);
// solveRCPSP_TT(22, 4, filename);
// solveRCPSP_TT(22, 7, filename);
// solveRCPSP_TT(23, 1, filename);
// solveRCPSP_TT(23, 4, filename);
// solveRCPSP_TT(26, 1, filename);
// solveRCPSP_TT(26, 4, filename);
// solveRCPSP_TT(26, 6, filename);
// solveRCPSP_TT(26, 7, filename);
// solveRCPSP_TT(26, 9, filename);
// solveRCPSP_TT(26, 10, filename);
// solveRCPSP_TT(27, 8, filename);
// solveRCPSP_TT(30, 2, filename);
// solveRCPSP_TT(30, 3, filename);
// solveRCPSP_TT(30, 4, filename);
// solveRCPSP_TT(30, 5, filename);
// solveRCPSP_TT(30, 8, filename);
// solveRCPSP_TT(30, 9, filename);
// solveRCPSP_TT(30, 10, filename);
// solveRCPSP_TT(31, 5, filename);
// solveRCPSP_TT(31, 7, filename);
// solveRCPSP_TT(32, 3, filename);
// solveRCPSP_TT(33, 3, filename);
// solveRCPSP_TT(33, 7, filename);
// solveRCPSP_TT(34, 8, filename);
// solveRCPSP_TT(37, 5, filename);
// solveRCPSP_TT(37, 8, filename);
// solveRCPSP_TT(37, 9, filename);
// solveRCPSP_TT(38, 1, filename);
// solveRCPSP_TT(38, 6, filename);
// solveRCPSP_TT(38, 7, filename);
// solveRCPSP_TT(38, 10, filename);
// solveRCPSP_TT(42, 2, filename);
// solveRCPSP_TT(42, 3, filename);
// solveRCPSP_TT(42, 6, filename);
// solveRCPSP_TT(42, 9, filename);
// solveRCPSP_TT(43, 2, filename);
// solveRCPSP_TT(43, 3, filename);
// solveRCPSP_TT(43, 5, filename);
// solveRCPSP_TT(43, 6, filename);
// solveRCPSP_TT(43, 8, filename);
// solveRCPSP_TT(43, 9, filename);
// solveRCPSP_TT(43, 10, filename);
// solveRCPSP_TT(44, 8, filename);
// solveRCPSP_TT(46, 2, filename);
// solveRCPSP_TT(46, 4, filename);
// solveRCPSP_TT(46, 5, filename);
// solveRCPSP_TT(46, 8, filename);
// solveRCPSP_TT(46, 9, filename);
// solveRCPSP_TT(46, 10, filename);
// solveRCPSP_TT(47, 1, filename);
// solveRCPSP_TT(47, 4, filename);
// solveRCPSP_TT(47, 5, filename);
// solveRCPSP_TT(47, 6, filename);
// solveRCPSP_TT(47, 10, filename);

    //solveRCPSP(-1, -1, filename);
    //solveRCPSP(3, 2, filename);
    //solveRCPSP(4, 3, filename);
    //solveRCPSP(4, 9, filename);

   //solveRCPSP(4, 7, filename);
    //solveRCPSP_Bi(8, 9, filename);
   // solveRCPSP(,10,filename);
  //
  //   solveRCPSP(34, 9, filename);
  //   solveRCPSP(34, 10, filename);
  // for(int i=35;i<49;i++) {
  //      for(int j=1;j<11;j++) {
  //      solveRCPSP(i,j,filename);
  //   //   getinitialHcost(i,j,filename);
  //      }
  //  }


    // for(int i=1;i<49;i++) {
    //      for(int j=1;j<11;j++) {
    //      solveRCPSP_TT(i,j,filename);
    //   //   getinitialHcost(i,j,filename);
    //      }
    //  }

//     solveRCPSP(1, 2, filename);
// solveRCPSP(1, 3, filename);
// solveRCPSP(1, 4, filename);
// solveRCPSP(1, 5, filename);
// solveRCPSP(1, 6, filename);
// solveRCPSP(1, 8, filename);
// solveRCPSP(1, 9, filename);
// solveRCPSP(1, 10, filename);
// solveRCPSP(2, 1, filename);
// solveRCPSP(2, 2, filename);
// solveRCPSP(2, 4, filename);
// solveRCPSP(2, 5, filename);
// solveRCPSP(2, 6, filename);
// solveRCPSP(2, 8, filename);
// solveRCPSP(2, 9, filename);
// solveRCPSP(2, 10, filename);
// solveRCPSP(3, 2, filename);
// solveRCPSP(3, 3, filename);
// solveRCPSP(3, 4, filename);
// solveRCPSP(3, 5, filename);
// solveRCPSP(3, 6, filename);
// solveRCPSP(3, 7, filename);
// solveRCPSP(3, 8, filename);
// solveRCPSP(3, 9, filename);
// solveRCPSP(4, 1, filename);
// solveRCPSP(4, 2, filename);
// solveRCPSP(4, 3, filename);
// solveRCPSP(4, 4, filename);
// solveRCPSP(4, 5, filename);
// solveRCPSP(4, 7, filename);
// solveRCPSP(4, 8, filename);
// solveRCPSP(4, 10, filename);
// solveRCPSP(5, 2, filename);
// solveRCPSP(5, 3, filename);
// solveRCPSP(5, 4, filename);
// solveRCPSP(5, 5, filename);
// solveRCPSP(5, 6, filename);
// solveRCPSP(5, 7, filename);
// solveRCPSP(5, 8, filename);
// solveRCPSP(5, 9, filename);
// solveRCPSP(6, 1, filename);
// solveRCPSP(6, 3, filename);
// solveRCPSP(6, 4, filename);
// solveRCPSP(6, 5, filename);
// solveRCPSP(6, 6, filename);
// solveRCPSP(6, 7, filename);
// solveRCPSP(6, 8, filename);
// solveRCPSP(6, 10, filename);
// solveRCPSP(7, 1, filename);
// solveRCPSP(7, 2, filename);
// solveRCPSP(7, 3, filename);
// solveRCPSP(7, 4, filename);
// solveRCPSP(7, 6, filename);
// solveRCPSP(7, 7, filename);
// solveRCPSP(7, 9, filename);
// solveRCPSP(7, 10, filename);
// solveRCPSP(8, 2, filename);
// solveRCPSP(8, 3, filename);
// solveRCPSP(8, 4, filename);
// solveRCPSP(8, 5, filename);
// solveRCPSP(8, 7, filename);
// solveRCPSP(8, 8, filename);
// solveRCPSP(8, 9, filename);
// solveRCPSP(8, 10, filename);
// solveRCPSP(9, 1, filename);
// solveRCPSP(9, 2, filename);
// solveRCPSP(9, 4, filename);
// solveRCPSP(9, 5, filename);
// solveRCPSP(9, 6, filename);
// solveRCPSP(9, 7, filename);
// solveRCPSP(9, 8, filename);
// solveRCPSP(9, 10, filename);
// solveRCPSP(10, 2, filename);
// solveRCPSP(10, 3, filename);
// solveRCPSP(10, 5, filename);
// solveRCPSP(10, 6, filename);
// solveRCPSP(10, 7, filename);
// solveRCPSP(10, 8, filename);
// solveRCPSP(10, 9, filename);
// solveRCPSP(10, 10, filename);
// solveRCPSP(11, 1, filename);
// solveRCPSP(11, 3, filename);
// solveRCPSP(11, 4, filename);
// solveRCPSP(11, 5, filename);
// solveRCPSP(11, 6, filename);
// solveRCPSP(11, 8, filename);
// solveRCPSP(11, 9, filename);
// solveRCPSP(11, 10, filename);
// solveRCPSP(12, 1, filename);
// solveRCPSP(12, 2, filename);
// solveRCPSP(12, 4, filename);
// solveRCPSP(12, 5, filename);
// solveRCPSP(12, 6, filename);
// solveRCPSP(12, 7, filename);
// solveRCPSP(12, 8, filename);
// solveRCPSP(12, 9, filename);
// solveRCPSP(13, 1, filename);
// solveRCPSP(13, 2, filename);
// solveRCPSP(13, 3, filename);
// solveRCPSP(13, 5, filename);
//     solveRCPSP(13, 7, filename);
//     solveRCPSP(13, 8, filename);
//     solveRCPSP(13, 9, filename);
//     solveRCPSP(13, 10, filename);
//
// solveRCPSP(1, 7, filename);
//  solveRCPSP(2, 3, filename);
//  solveRCPSP(2, 7, filename);
//  solveRCPSP(3, 1, filename);
//  solveRCPSP(3, 10, filename);
//  solveRCPSP(4, 6, filename);
//  solveRCPSP(4, 9, filename);
 // solveRCPSP(5, 1, filename);
//  solveRCPSP(5, 10, filename);
 // solveRCPSP(6, 2, filename);
 // solveRCPSP(6, 9, filename);
 // solveRCPSP(7, 5, filename);
 // solveRCPSP(7, 8, filename);
 // solveRCPSP(8, 1, filename);
 // solveRCPSP(8, 6, filename);
 // solveRCPSP(9, 3, filename);
 // solveRCPSP(9, 9, filename);
 // solveRCPSP(10, 1, filename);
 // solveRCPSP(10, 4, filename);
 // solveRCPSP(11, 2, filename);
 // solveRCPSP(11, 7, filename);
 // solveRCPSP(12, 3, filename);
 // solveRCPSP(12, 10, filename);
 // solveRCPSP(13, 4, filename);
 // solveRCPSP(13, 6, filename);
 // solveRCPSP(14, 2, filename);
 // solveRCPSP(14, 8, filename);
 // solveRCPSP(15, 5, filename);
 // solveRCPSP(15, 9, filename);
 // solveRCPSP(16, 2, filename);
 // solveRCPSP(16, 7, filename);
 // solveRCPSP(17, 8, filename);
 // solveRCPSP(17, 10, filename);
 // solveRCPSP(18, 4, filename);
 // solveRCPSP(18, 7, filename);
 // solveRCPSP(19, 1, filename);
 // solveRCPSP(19, 5, filename);
 // solveRCPSP(20, 1, filename);
 // solveRCPSP(20, 5, filename);
 // solveRCPSP(21, 1, filename);
 // solveRCPSP(21, 9, filename);
 // solveRCPSP(22, 4, filename);
 // solveRCPSP(22, 6, filename);
 // solveRCPSP(23, 2, filename);
 // solveRCPSP(23, 5, filename);
 // solveRCPSP(24, 3, filename);
 // solveRCPSP(24, 6, filename);
 // solveRCPSP(25, 2, filename);
 // solveRCPSP(25, 10, filename);
 // solveRCPSP(26, 4, filename);
 // solveRCPSP(26, 7, filename);
 // solveRCPSP(27, 6, filename);
 // solveRCPSP(27, 8, filename);
 // solveRCPSP(28, 3, filename);
 // solveRCPSP(28, 9, filename);
 // solveRCPSP(29, 1, filename);
 // solveRCPSP(29, 6, filename);
 // solveRCPSP(30, 2, filename);
 // solveRCPSP(30, 4, filename);
 // solveRCPSP(31, 2, filename);
 // solveRCPSP(31, 9, filename);
 // solveRCPSP(32, 6, filename);
 // solveRCPSP(32, 10, filename);
 // solveRCPSP(33, 1, filename);
 // solveRCPSP(33, 7, filename);
 // solveRCPSP(34, 5, filename);
 // solveRCPSP(34, 9, filename);
 // solveRCPSP(35, 1, filename);
 // solveRCPSP(35, 6, filename);
 // solveRCPSP(36, 1, filename);
 // solveRCPSP(36, 5, filename);
 // solveRCPSP(37, 5, filename);
 // solveRCPSP(37, 8, filename);
 // solveRCPSP(38, 1, filename);
 // solveRCPSP(38, 4, filename);
 // solveRCPSP(39, 2, filename);
 // solveRCPSP(39, 7, filename);
 // solveRCPSP(40, 2, filename);
 // solveRCPSP(40, 7, filename);
 // solveRCPSP(41, 4, filename);
 // solveRCPSP(41, 10, filename);
 // solveRCPSP(42, 1, filename);
 // solveRCPSP(42, 8, filename);
 // solveRCPSP(43, 3, filename);
 // solveRCPSP(43, 7, filename);
 // solveRCPSP(44, 3, filename);
 // solveRCPSP(44, 6, filename);
 // solveRCPSP(45, 1, filename);
 // solveRCPSP(45, 5, filename);
 // solveRCPSP(46, 1, filename);
 // solveRCPSP(46, 8, filename);
 // solveRCPSP(47, 3, filename);
 // solveRCPSP(47, 6, filename);
 // solveRCPSP(48, 2, filename);
 // solveRCPSP(48, 9, filename);
return;;

}

void getinitialHcost(int group, int exam, const std::string &filename) {
    std::cout << "started solving: " << group<<":"<<exam << std::endl;
    count=0;
    double initalHcost=0;
    getPetri(petri, group, exam);
    getRCPSP(RCPSPex, group, exam);

    RCPSPState first;
    initalHcost=getForwardHcost(first.unstartedTransitions,first.activeTransitionIndices);

    std::cout << "initalHcost\n";



    std::ofstream file(filename, std::ios::app);
    file << group << "," << exam << "," << initalHcost<<std::endl;

}

