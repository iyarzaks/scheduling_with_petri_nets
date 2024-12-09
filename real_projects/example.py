import math
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class Place:
    id: str
    tokens: int = 0


@dataclass
class Transition:
    id: str
    duration: int
    inputs: List[Place]
    outputs: List[Place]


class PetriNet:
    def __init__(self):
        self.places: Dict[str, Place] = {}
        self.transitions: Dict[str, Transition] = {}
        self.final_place: Place = None

    def add_place(self, place_id: str) -> Place:
        place = Place(place_id)
        self.places[place_id] = place
        return place

    def add_transition(
        self, trans_id: str, duration: int, inputs: List[str], outputs: List[str]
    ) -> Transition:
        input_places = [self.places[p] for p in inputs]
        output_places = [self.places[p] for p in outputs]
        transition = Transition(trans_id, duration, input_places, output_places)
        self.transitions[trans_id] = transition
        return transition

    def set_final_place(self, place_id: str):
        self.final_place = self.places[place_id]


class PetriNetHeuristic:
    def __init__(self, petri_net: PetriNet):
        self.net = petri_net
        # Cache for memoization
        self.min_duration_cache: Dict[str, int] = {}

    def find_and_joins(self) -> List[Transition]:
        """Find all AND-join transitions (transitions with multiple inputs)."""
        return [t for t in self.net.transitions.values() if len(t.inputs) > 1]

    def get_predecessors(self, place: Place) -> List[Transition]:
        """Get all transitions that output to the given place."""
        return [t for t in self.net.transitions.values() if place in t.outputs]

    def get_min_duration_to_place(self, place: Place, visited: Set[str] = None) -> int:
        """Calculate minimum duration to reach a place from any current marking."""
        if visited is None:
            visited = set()

        # Check cache
        if place.id in self.min_duration_cache:
            return self.min_duration_cache[place.id]

        # Base case: if place has tokens, duration is 0
        if place.tokens > 0:
            return 0

        # Avoid cycles
        if place.id in visited:
            return math.inf

        visited.add(place.id)

        # Get all predecessor transitions
        pred_transitions = self.get_predecessors(place)
        if not pred_transitions:
            return math.inf

        # For each predecessor transition, calculate:
        # transition_duration + max(min_duration to reach each input place)
        min_duration = math.inf
        for trans in pred_transitions:
            input_durations = [
                self.get_min_duration_to_place(p, visited.copy()) for p in trans.inputs
            ]
            if all(d != math.inf for d in input_durations):
                path_duration = trans.duration + max(input_durations)
                min_duration = min(min_duration, path_duration)

        # Cache result
        self.min_duration_cache[place.id] = min_duration
        return min_duration

    def get_sequential_duration(self, start_trans: Transition) -> int:
        """Calculate total duration of sequential path from a transition to final place."""
        current = start_trans
        total_duration = current.duration

        while True:
            # Get the next transition in sequence
            next_place = current.outputs[0]  # Assuming sequential path
            next_transitions = [
                t for t in self.net.transitions.values() if next_place in t.inputs
            ]

            if not next_transitions:
                break

            current = next_transitions[0]
            total_duration += current.duration

            if self.net.final_place in current.outputs:
                break

        return total_duration

    def calculate_heuristic(self, marking: Dict[str, int]) -> int:
        """Calculate the heuristic value for the given marking."""
        # Update tokens based on marking
        for place_id, tokens in marking.items():
            self.net.places[place_id].tokens = tokens

        # Clear cache for new calculation
        self.min_duration_cache.clear()

        # Find AND-joins
        and_joins = self.find_and_joins()

        if not and_joins:
            # If no AND-joins, calculate direct path to final place
            return self.get_min_duration_to_place(self.net.final_place)

        # For each AND-join:
        # 1. Calculate min duration to reach each input
        # 2. Take maximum since all inputs must be reached
        # 3. Add sequential path duration to final place
        min_total_duration = math.inf

        for join in and_joins:
            # Calculate parallel portion (max of minimums to reach each input)
            parallel_durations = [
                self.get_min_duration_to_place(input_place)
                for input_place in join.inputs
            ]
            if any(d == math.inf for d in parallel_durations):
                continue

            parallel_duration = max(parallel_durations)

            # Calculate sequential portion
            sequential_duration = self.get_sequential_duration(join)

            # Total duration for this path
            total_duration = parallel_duration + sequential_duration
            min_total_duration = min(min_total_duration, total_duration)

        return min_total_duration if min_total_duration != math.inf else 0


# Example usage:
def create_example_net() -> PetriNet:
    net = PetriNet()

    # Add places
    places = [
        "initial_place",
        "p_4",
        "p_5",
        "p_6",
        "p_7",
        "p_8",
        "p_9",
        "p_10",
        "p_11",
        "p_12",
        "final_place",
    ]
    for p in places:
        net.add_place(p)

    # Add transitions
    net.add_transition("tauSplit_1", 0, ["initial_place"], ["p_4", "p_6", "p_8"])
    net.add_transition("a", 3, ["p_4"], ["p_5"])
    net.add_transition("b", 7, ["p_6"], ["p_7"])
    net.add_transition("skip_3", 0, ["p_6"], ["p_7"])
    net.add_transition("c", 8, ["p_8"], ["p_10"])
    net.add_transition("skip_4", 0, ["p_8"], ["p_10"])
    net.add_transition("d", 2, ["p_10"], ["p_9"])
    net.add_transition("e", 30, ["p_5", "p_7", "p_9"], ["p_11"])
    net.add_transition("f", 19, ["p_11"], ["p_12"])
    net.add_transition("g", 7, ["p_12"], ["final_place"])

    net.set_final_place("final_place")
    return net


# Test the implementation
net = create_example_net()
heuristic = PetriNetHeuristic(net)
initial_marking = {"initial_place": 1}
h_value = heuristic.calculate_heuristic(initial_marking)
print(f"Heuristic value for initial marking: {h_value}")  # Should print 67
