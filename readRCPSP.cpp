#include <iostream>
#include <fstream>
#include "json.hpp"  // Includes the json.hpp file
#include "petriclasses.h"
using json = nlohmann::json;  // alias for nlohmann::json

int main() {
    // Open the file for reading
    RCPSP_example rcpsp_example;
    std::ifstream input_file("rcpspExample.json");

    // If the file could not be opened
    if (!input_file.is_open()) {
        std::cerr << "Failed to open rcpspExample.json" << std::endl;
        return 1;
    }

    // Create JSON object
    json j;

    // Read into the JSON object
    input_file >> j;

    // Check if the JSON is an array and contains at least one part
    if (j.is_array() && j.size() > 0) {
        // Print the names of the activities in Part 1 (first 32 activities)
        std::cout << "Part 1 (Activities):" << std::endl;

        // Loop over the first 32 elements (0-31)
        for (int i = 0; i < j.size()-3 && i < j.size(); ++i) {
            const auto& activity1 = j[i];
            Activity activity(activity1["duration"], activity1["name"], activity1["resource_demands"]);
            rcpsp_example.addActivity(activity);
            // Check if the "name" and "duration" fields exist in the current activity
            if (activity1.contains("name") && activity1.contains("duration")) {
                std::cout << "Name: " << activity1["name"] << std::endl;
                std::cout << "Duration: " << activity1["duration"] << std::endl;

                // Print resource demands, if they exist
                if (activity1.contains("resource_demands") && !activity1["resource_demands"].empty()) {
                    std::cout << "Resource Demands:" << std::endl;
                    for (auto& demand : activity1["resource_demands"].items()) {
                        std::cout << "  " << demand.key() << ": " << demand.value() << std::endl;
                    }
                } else {
                    std::cout << "No resource demands." << std::endl;
                }
            }
            std::cout << std::endl;  // Space between activities
        }

    } else {
        std::cerr << "The JSON structure is invalid or does not have enough parts." << std::endl;
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
        std::cout << "Key: " << key << " -> Values: ";
        for (const auto& val : value) {
            std::cout << val << " ";
            rcpsp_example.backword_dependencies[key-1].push_back(val);
        }
        std::cout << std::endl;
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
        std::cout << "Key: " << key << " -> Values: ";
        for (const auto& val : value) {
            std::cout << val << " ";

            rcpsp_example.dependencies[key-1].push_back(val);

        }
        std::cout << std::endl;
    }
    // Close the file after reading
    //std::cout << "Current size of dependencies: " << rcpsp_example.dependencies.size() << std::endl;

    const auto& resources_json = j[34];  // Get the resources part

    // Loop through and add resources individually to the class
    for (auto& [key, value] : resources_json.items()) {
        // Assuming the value is always an integer (resources are integers)
        rcpsp_example.addResource(key, value.get<int>());
    }
    return 0;
}