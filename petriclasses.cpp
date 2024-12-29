#include <string>
#include <map>
#include <vector>
class Activity {
public:
    int duration;
    int early_finish;
    int early_start;
    int late_finish;
    int late_start;
    std::string name;
    std::map<std::string, int> resource_demands; // For resources like 'R1': 3

    // Constructor
    Activity(int dur, const std::string& n, const std::map<std::string, int>& resources)
        : duration(dur), name(n), resource_demands(resources), early_finish(0), early_start(0), late_finish(0), late_start(0) {}
    ~Activity() {};
};

class RCPSP_example{
  public:
    //for better time array insted of vec?
    std::vector<Activity> activities; // List of Activity objects
    int activity_len;
    // Add an activity
    void addActivity(const Activity& activity) {
        activities.push_back(activity);
    }
   //skip on activity name duration as it can be acssesd with acvivity[num].duration
    std::vector<std::vector<int>> dependencies;
    std::vector<std::vector<int>> backword_dependencies;
    void addDependencies(std::vector<std::vector<int>>& dependencieslist,std::vector<std::vector<int>> dependencies) {
      for (int i = 0; i < activity_len; i++) {
        dependencies.push_back(dependencies[i]);
      }
    }
    void addBackword_dependencies(std::vector<std::vector<int>>& backword_dependencies,std::vector<std::vector<int>> dependencies) {
        for (int i = 0; i < activity_len; i++) {
            backword_dependencies.push_back(dependencies[i]);
        }
    }
    std::vector<int> recsores;
    std::vector<std::map<std::string, int>> resources;

    // Create the first map and add it to the vector
    std::map<std::string, int> resource1;
    void addResources(const std::map<std::string, int>& resources) {
      for (int i = 0; i < resources.size(); i++) {

      }
    }
    ~RCPSP_example() {}
    RCPSP_example(){};
    //didnt put activity_names_duration activity_names_set depenedncy_deep_set
};
class Place {
public:
    std::map<std::string, int> arcs_in;  // Arcs coming into the place
    std::map<std::string, int> arcs_out; // Arcs going out from the place
    int duration;                       // Duration of the place
    std::string name;                   // Name of the place
    std::vector<int> state;             // State values

    ~Place() {};
    // Constructor
    Place(const std::string& placeName,
          const std::map<std::string, int>& inputArcs = {},
          const std::map<std::string, int>& outputArcs = {},
          const std::vector<int>& initialState = {},
          int initialDuration = 0)
        : name(placeName), arcs_in(inputArcs), arcs_out(outputArcs), state(initialState), duration(initialDuration) {}

};
class Place_dict {
public:
    std::map<std::string, int> arcs_in;  // Arcs coming into the place
    std::map<std::string, int> arcs_out; // Arcs going out from the place
    int duration;                       // Duration of the place
    std::string name;                   // Name of the place
    std::vector<int> state;             // State values

    // Constructor
    ~Place_dict() {};
    Place_dict(const std::string& placeName,
          const std::map<std::string, int>& inputArcs = {},
          const std::map<std::string, int>& outputArcs = {},
          const std::vector<int>& initialState = {},
          int initialDuration = 0)
        : name(placeName), arcs_in(inputArcs), arcs_out(outputArcs), state(initialState), duration(initialDuration) {}
};
class Transition {
public:
    std::map<std::string, int> arcs_in;  // Arcs coming into the place
    std::map<std::string, int> arcs_out; // Arcs going out from the place
    int duration;                       // Duration of the place
    std::string name;                   // Name of the place

    // Constructor
    ~Transition() {};
    Transition(const std::string& TransitionName,
          const std::map<std::string, int>& inputArcs = {},
          const std::map<std::string, int>& outputArcs = {},
          const std::vector<int>& initialState = {},
          int initialDuration = 0)
        : name(TransitionName), arcs_in(inputArcs), arcs_out(outputArcs), duration(initialDuration) {}
};
class Transition_dict {
public:
    std::map<std::string, int> arcs_in;  // Arcs coming into the place
    std::map<std::string, int> arcs_out; // Arcs going out from the place
    int duration;                       // Duration of the place
    std::string name;                   // Name of the place

    // Constructor
    ~Transition_dict() {};
    Transition_dict(const std::string& Transition_dictName,
          const std::map<std::string, int>& inputArcs = {},
          const std::map<std::string, int>& outputArcs = {},
          const std::vector<int>& initialState = {},
          int initialDuration = 0)
        : name(Transition_dictName), arcs_in(inputArcs), arcs_out(outputArcs), duration(initialDuration) {}
};
class PetriExample {
public:
  ~PetriExample() {};
  PetriExample(){};
    std::vector<Place> places;  // Vector of Place objects

    // Add a place to the Petri example
    void addPlace(const Place& place) {
        places.push_back(place);
    }
    int place_len;
    std::vector<Place_dict> places_dict;  // Vector of Place objects

    void addPlace_dict(const Place_dict& place_dict) {
        places_dict.push_back(place_dict);
    }
    std::vector<Transition> Transitions;  // Vector of Place objects
    void addTransition(const Transition& transition) {
        Transitions.push_back(transition);
    }
    std::vector<Transition_dict> Transitions_dict;  // Vector of Place objects
    void addTransition_dist(const Transition_dict& transition_dict) {
        Transitions_dict.push_back(transition_dict);
    }


};