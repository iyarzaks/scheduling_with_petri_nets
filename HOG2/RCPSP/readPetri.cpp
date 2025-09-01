#include <iostream>
#include <fstream>
#include "json.hpp"

using json = nlohmann::json;  // alias for nlohmann::json

void getPetri(PetriExample& petriExample, int group, int exam, const std::string &instanceType = "j30") {
    petriExample.reset();
    std::string folderName;

    // Decide base path depending on instance type
    std::string basePath;
    if (instanceType == "j30") {basePath = "json_outputs_j30/j30";}
    else if (instanceType == "j60") {basePath = "json_outputs_j60/j60";}
    else if (instanceType == "j90") {basePath = "json_outputs_j90/j90";}
    else if (instanceType == "j120") {basePath = "json_outputs_j120/j120";}
    else {
        throw std::runtime_error("Unsupported instance type: " + instanceType);
    }

    // Handle special case group == -1
    if (group == -1) {
        folderName = basePath + "/smallData";
    } else {
        folderName = basePath + std::to_string(group) + "_" + std::to_string(exam);
    }

    // Try opening a file (example: rcpsp.json or petriExample.json)
    // std::ifstream test_file(folderName + "/petri.json");
    // if (!test_file) {
    //     throw std::runtime_error("Could not open: " + folderName + "/petri.json");
    // }


    //std::ifstream input_file("rcpspExample.json");
    std::ifstream input_file(folderName+"/petri.json");
    // If the file could not be opened
    if (!input_file.is_open()) {
        std::cerr << "Failed to open petri.json" << std::endl;
        return;
    }
    // Create JSON object
    json j;

    // Read into the JSON object
    input_file >> j;
    int placeSize=j[0];
    int placedictSize=j[placeSize+1];
    int tranSize=j[placedictSize+1+placeSize+1];
    int trandictSize=j[tranSize+1+placedictSize+1+placeSize+1];

    for (int i = 1; i < placeSize+1; i++) {
        std::vector<std::vector<int>> state;
        if (j[i].size() == 4) {

            state.emplace_back(std::vector<int>{0});
        }
        else {
            state=j[i]["state"];
        }
        Place place(j[i]["name"],j[i]["arcs_in"],j[i]["arcs_out"],state,j[i]["duration"]);
        petriExample.places.push_back(place);

    }
    for (int i = placeSize+2; i < placedictSize+1+placeSize+1; i++) {
        std::vector<std::vector<int>> state;
        if (j[i].size() == 4) {

            state.emplace_back(std::vector<int>{0});
        }
        else {
            state=j[i]["state"];
        }
        Place_dict place_dict(j[i]["name"],j[i]["arcs_in"],j[i]["arcs_out"],state,j[i]["duration"]);
        petriExample.places_dict.push_back(place_dict);

    }
    for (int i = placedictSize+1+placeSize+1+1; i < placedictSize+1+placeSize+1+1+tranSize; i++) {

        Transition tran(j[i]["name"],j[i]["arcs_in"],j[i]["arcs_out"],j[i]["duration"]);
        petriExample.Transitions.push_back(tran);

    }
    for (int i = placedictSize+1+placeSize+1+tranSize+1+1; i < placedictSize+1+placeSize+1+1+tranSize+trandictSize+1; i++) {

        Transition_dict tran_dict(j[i]["name"],j[i]["arcs_in"],j[i]["arcs_out"],j[i]["duration"]);
        petriExample.Transitions_dict.push_back(tran_dict);

    }

}
void getRCPSP(RCPSP_example& rcpsp_example,int group,int exam, const std::string &instanceType = "j30") {
    // Open the file for reading
    rcpsp_example.reset();
    std::string folderName;
    std::string basePath;
    if (instanceType == "j30") {basePath = "json_outputs_j30/j30";}
    else if (instanceType == "j60") {basePath = "json_outputs_j60/j60";}
    else if (instanceType == "j90") {basePath = "json_outputs_j90/j90";}
    else if (instanceType == "j120") {basePath = "json_outputs_j120/j120";}
    else {
        throw std::runtime_error("Unsupported instance type: " + instanceType);
    }

    // Handle special case group == -1
    if (group == -1) {
        folderName = basePath + "/smallData";
    } else {
        folderName = basePath + std::to_string(group) + "_" + std::to_string(exam);
    }

    // Try opening a file (example: rcpsp.json or petriExample.json)
    // std::ifstream test_file(folderName + "/petri.json");
    // if (!test_file) {
    //     throw std::runtime_error("Could not open: " + folderName + "/petri.json");
    // }


    //std::ifstream input_file("rcpspExample.json");
    std::ifstream input_file(folderName+"/rcpsp.json");
    // If the file could not be opened
    if (!input_file.is_open()) {
        std::cerr << "Failed to open rcpsp.json" << std::endl;
        return;
    }

    // Create JSON object
    json j;

    // Read into the JSON object
    input_file >> j;

    // Check if the JSON is an array and contains at least one part
    if (j.is_array() && j.size() > 0) {
        // Print the names of the activities in Part 1 (first 32 activities)
int size= j.size();
        // Loop over the first 32 elements (0-31)
        for (int i = 0; i < j.size()-4 && i < j.size(); ++i) {
            const auto& activity1 = j[i];
            Activity activity(activity1["duration"], activity1["name"], activity1["resource_demands"]);
            rcpsp_example.addActivity(activity);
            // Check if the "name" and "duration" fields exist in the current activity
            if (activity1.contains("name") && activity1.contains("duration")) {


                // Print resource demands, if they exist
                if (activity1.contains("resource_demands") && !activity1["resource_demands"].empty()) {

                    for (auto& demand : activity1["resource_demands"].items()) {
                        //std::cout << "  " << demand.key() << ": " << demand.value() << std::endl;
                    }
                } else {
                    //std::cout << "No resource demands." << std::endl;
                }
            }
            //std::cout << std::endl;  // Space between activities
        }

    } else {
        //std::cerr << "The JSON structure is invalid or does not have enough parts." << std::endl;
    }
    rcpsp_example.dependencies.resize(j.size()-4);
    rcpsp_example.backword_dependencies.resize(j.size()-4);
    std::vector<std::pair<int, json>> sorted32;
    for (const auto& item : j[j.size()-4].items()) {
        sorted32.push_back({std::stoi(item.key()), item.value()});
    }

    // Sort by the integer value of the keys
    std::sort(sorted32.begin(), sorted32.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Print the sorted keys and their associated values
    for (const auto& [key, value] : sorted32) {
        //std::cout << "Key: " << key << " -> Values: ";
        for (const auto& val : value) {
            //std::cout << val << " ";
            rcpsp_example.backword_dependencies[key-1].push_back(val);
        }
        //std::cout << std::endl;
    }
    std::vector<std::pair<int, json>> sorted33;
    for (const auto& item : j[j.size()-3].items()) {
        sorted33.push_back({std::stoi(item.key()), item.value()});
    }

    // Sort by the integer value of the keys
    std::sort(sorted33.begin(), sorted33.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Print the sorted keys and their associated values
    for (const auto& [key, value] : sorted33) {
       // std::cout << "Key: " << key << " -> Values: ";
        for (const auto& val : value) {
            //std::cout << val << " ";

            rcpsp_example.dependencies[key-1].push_back(val);

        }
        //std::cout << std::endl;
    }
    // Close the file after reading
    //std::cout << "Current size of dependencies: " << rcpsp_example.dependencies.size() << std::endl;

    const auto& resources_json = j[j.size()-1];  // Get the resources part

    // Loop through and add resources individually to the class
    for (auto& [key, value] : resources_json.items()) {
        // Assuming the value is always an integer (resources are integers)
        rcpsp_example.addResource(key, value.get<int>());
    }
    const auto& deep_set_array = j[j.size() - 2];

    if (deep_set_array.is_array()) {
        for (const auto& pair : deep_set_array) {
            if (pair.is_array() && pair.size() == 2) {
                std::string from = pair[0];
                std::string to = pair[1];
                rcpsp_example.deep_dependencies.insert({from, to});
            } else {
                std::cerr << "Malformed deep dependency pair in JSON." << std::endl;
            }
        }
    } else {
        std::cerr << "Expected an array for deep dependencies." << std::endl;
    }
    return;
}


void getRCPSP_old(RCPSP_example& rcpsp_example,int group,int exam) {
    // Open the file for reading
    rcpsp_example.reset();
    std::string folderName;
    if (group==-1) {
        folderName = "json_outputs_old/smallData";;
    }
    else {
        std::string basePath = "json_outputs_old/j30";
        folderName = basePath + std::to_string(group) + "_" + std::to_string(exam);
    }
    std::ifstream input_file(folderName+"/rcpsp.json");

    // If the file could not be opened
    if (!input_file.is_open()) {
        std::cerr << "Failed to open rcpsp.json" << std::endl;
        return;
    }
    // Create JSON object
    json j;

    // Read into the JSON object
    input_file >> j;

    // Check if the JSON is an array and contains at least one part
    if (j.is_array() && j.size() > 0) {
        // Print the names of the activities in Part 1 (first 32 activities)

        // Loop over the first 32 elements (0-31)
        for (int i = 0; i < j.size()-3 && i < j.size(); ++i) {
            const auto& activity1 = j[i];
            Activity activity(activity1["duration"], activity1["name"], activity1["resource_demands"]);
            rcpsp_example.addActivity(activity);
            // Check if the "name" and "duration" fields exist in the current activity
            if (activity1.contains("name") && activity1.contains("duration")) {


                // Print resource demands, if they exist
                if (activity1.contains("resource_demands") && !activity1["resource_demands"].empty()) {

                    for (auto& demand : activity1["resource_demands"].items()) {
                        //std::cout << "  " << demand.key() << ": " << demand.value() << std::endl;
                    }
                } else {
                    //std::cout << "No resource demands." << std::endl;
                }
            }
            //std::cout << std::endl;  // Space between activities
        }

    } else {
        //std::cerr << "The JSON structure is invalid or does not have enough parts." << std::endl;
    }
    rcpsp_example.dependencies.resize(j.size()-3);
    rcpsp_example.backword_dependencies.resize(j.size()-3);
    std::vector<std::pair<int, json>> sorted32;
    for (const auto& item : j[j.size()-3].items()) {
        sorted32.push_back({std::stoi(item.key()), item.value()});
    }

    // Sort by the integer value of the keys
    std::sort(sorted32.begin(), sorted32.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Print the sorted keys and their associated values
    for (const auto& [key, value] : sorted32) {
        //std::cout << "Key: " << key << " -> Values: ";
        for (const auto& val : value) {
            //std::cout << val << " ";
            rcpsp_example.backword_dependencies[key-1].push_back(val);
        }
        //std::cout << std::endl;
    }
    std::vector<std::pair<int, json>> sorted33;
    for (const auto& item : j[j.size()-2].items()) {
        sorted33.push_back({std::stoi(item.key()), item.value()});
    }

    // Sort by the integer value of the keys
    std::sort(sorted33.begin(), sorted33.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Print the sorted keys and their associated values
    for (const auto& [key, value] : sorted33) {
       // std::cout << "Key: " << key << " -> Values: ";
        for (const auto& val : value) {
            //std::cout << val << " ";

            rcpsp_example.dependencies[key-1].push_back(val);

        }
        //std::cout << std::endl;
    }
    // Close the file after reading
    //std::cout << "Current size of dependencies: " << rcpsp_example.dependencies.size() << std::endl;

    const auto& resources_json = j[34];  // Get the resources part

    // Loop through and add resources individually to the class
    for (auto& [key, value] : resources_json.items()) {
        // Assuming the value is always an integer (resources are integers)
        rcpsp_example.addResource(key, value.get<int>());
    }
    return;
}