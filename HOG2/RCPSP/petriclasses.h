#include <string>
#include <map>
#include <vector>
#include <unordered_map>
class Activity {
public:
    int duration;
    int early_finish;
    int early_start;
    int late_finish;
    int late_start;
    std::string name;
    std::map<std::string, int> resource_demands; // For resources like 'R1': 3
    Activity(): duration(0), early_finish(0), early_start(0), late_finish(0), late_start(0) {
    };
    // Constructor
    Activity(int dur, const std::string& n, const std::map<std::string, int>& resources)
        : duration(dur), name(n), resource_demands(resources), early_finish(0), early_start(0), late_finish(0), late_start(0) {}
    ~Activity() = default;
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
    std::vector<std::vector<std::string>> dependencies;
    std::vector<std::vector<std::string>> backword_dependencies;

    std::vector<std::pair<std::string, int>> resources;

    // Function to add a single resource to the vector
    void addResource(const std::string& name, int value) {
        resources.push_back({name, value});  // Push as a pair
    }
    ~RCPSP_example() = default;
    RCPSP_example(): activity_len(0) {
    };
    //didnt put activity_names_duration activity_names_set depenedncy_deep_set
};



class Place {
public:
    std::string name;
    std::map<std::string, int> arcs_in;
    std::map<std::string, int> arcs_out;
    std::vector<std::vector<int>> state;
    int duration;             // TODO delete

    ~Place() {};
    // Constructor
    Place(const std::string& placeName,
          const std::map<std::string, int>& inputArcs = {},
          const std::map<std::string, int>& outputArcs = {},
          const std::vector<std::vector<int>>& initialState = {},
          int initialDuration = 0)
        : name(placeName), arcs_in(inputArcs), arcs_out(outputArcs), state(initialState), duration(initialDuration) {}

};
class Place_dict {
public:
    std::map<std::string, int> arcs_in;  // Arcs coming into the place
    std::map<std::string, int> arcs_out; // Arcs going out from the place
    int duration;                       // Duration of the place
    std::string name;                   // Name of the place
    std::vector<std::vector<int>> state;            // State values

    // Constructor
    ~Place_dict() {};
    Place_dict(const std::string& placeName,
          const std::map<std::string, int>& inputArcs = {},
          const std::map<std::string, int>& outputArcs = {},
          const std::vector<std::vector<int>>& initialState = {},
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

          int initialDuration = 0)
        : name(TransitionName), arcs_in(inputArcs), arcs_out(outputArcs), duration(initialDuration) {}
    bool operator==(const Transition& other) const {
        return name == other.name &&
               arcs_in == other.arcs_in &&
               arcs_out == other.arcs_out &&
               duration == other.duration;
    }
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
          int initialDuration = 0)
        : name(Transition_dictName), arcs_in(inputArcs), arcs_out(outputArcs), duration(initialDuration) {}
};
class PetriExample {
public:
  ~PetriExample() = default;
  PetriExample(): place_len(0) {
  };
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