#include <iostream>
#include <fstream>
#include "json.hpp"  // Includes the json.hpp file
#include "petriclasses.h"
using json = nlohmann::json;  // alias for nlohmann::json

int main() {
    // Open the file for reading
    RCPSP_example rcpsp_example;
    std::ifstream input_file("petriExample.json");

    // If the file could not be opened
    if (!input_file.is_open()) {
        std::cerr << "Failed to open petriExample.json" << std::endl;
        return 1;
    }

    // Create JSON object
    json j;

    // Read into the JSON object
    input_file >> j;
    int placeSize=j[0];
    int placedictSize=j[placeSize+1];
    int tranSize=j[placedictSize+1+placeSize+1];
    int trandictSize=j[tranSize+1+placedictSize+1+placeSize+1];
    PetriExample petriExample;
    for (int i = 1; i < placeSize+1; i++) {
        std::vector<std::vector<int>> state;
        if (j[i].size() == 4) {

            state.push_back(std::vector<int>(1));
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

            state.push_back(std::vector<int>(1));
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


    return 0;
}