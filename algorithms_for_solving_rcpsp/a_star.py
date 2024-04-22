import copy
import heapq

from tqdm import tqdm

from RCPSP_modeling.rcpsp_petri_net import (
    PlaceTimePetriNetTransition,
    TIME,
    COUNT,
    FINISH,
    START,
)
from RCPSP_modeling.rcpsp_petri_net import RcpspPlaceTimePetriNet


class RgNode:
    def __init__(
        self,
        petri_net_to_solve,
        heuristic_function,
        rcpsp_base,
        previous_rg_node=None,
        transition: PlaceTimePetriNetTransition = None,
    ):
        self.marking = {}
        if previous_rg_node is None:
            self.finished_activities = {}
            self.started_activities = {}
            for place in petri_net_to_solve.places:
                if place.state is not None:
                    self.marking[place.name] = place.state
        else:
            self.update_marking_and_activities(
                previous_rg_node.marking,
                previous_rg_node.finished_activities,
                previous_rg_node.started_activities,
                transition,
                petri_net_to_solve,
            )
        self.available_transitions = [
            transition
            for transition in petri_net_to_solve.transitions
            if transition.is_available(self.marking)
        ]
        if len(self.finished_activities) > 0:
            self.g_score = max(self.finished_activities.values())
        else:
            self.g_score = 0
        self.h_score = heuristic_function(rcpsp_base, self.finished_activities)
        self.f_score = self.g_score + self.h_score

    def update_marking_and_activities(
        self,
        previous_marking,
        previous_finished_activities,
        previous_started_activities,
        transition: PlaceTimePetriNetTransition,
        petri_net_to_solve: RcpspPlaceTimePetriNet,
    ):
        new_marking = previous_marking.copy()
        new_finished_activities = previous_finished_activities.copy()
        new_started_activities = previous_started_activities.copy()
        firing_time = max(previous_marking[p].get(TIME, 0) for p in transition.arcs_in)
        for out_node in transition.arcs_out:
            new_marking[out_node] = {
                TIME: firing_time + petri_net_to_solve.places_dict[out_node].duration,
                COUNT: previous_marking[out_node].get(COUNT, 0)
                + transition.arcs_out[out_node],
            }
        for in_node in transition.arcs_in:
            new_marking[in_node] = {
                TIME: previous_marking[in_node].get(TIME, 0),
                COUNT: previous_marking[in_node].get(COUNT, 0)
                - transition.arcs_in[in_node],
            }
        if FINISH in transition.name:
            new_finished_activities[transition.name.replace(FINISH, "")] = firing_time
        if START in transition.name:
            new_started_activities[transition.name.replace(START, "")] = firing_time
        self.finished_activities = new_finished_activities
        self.started_activities = new_started_activities
        self.marking = new_marking

    def __lt__(self, other):
        return self.f_score < other.f_score

    def __eq__(self, other):
        return self.f_score == other.f_score


class AStarSolver:
    def __init__(
        self,
        petri_net_to_solve: RcpspPlaceTimePetriNet,
        rcpsp_base,
        heuristic_function,
    ):
        self.petri_net_to_solve = petri_net_to_solve
        self.rcpsp_base = rcpsp_base
        self.heuristic_function = heuristic_function
        self.start_node = RgNode(petri_net_to_solve, heuristic_function, rcpsp_base)
        print(self.start_node)

    def is_final_node(self, node: RgNode) -> bool:
        final_places = [
            place
            for place in self.petri_net_to_solve.places
            if len(place.arcs_out) == 0
        ]
        if all(node.marking[place.name].get("count", 0) > 0 for place in final_places):
            return True
        else:
            return False

    def solve(self):
        progress_bar = tqdm(total=1000000, desc="Processing")
        visited = 0
        current = self.start_node
        available = []
        while not self.is_final_node(current):
            available_transitions = current.available_transitions
            for transition in available_transitions:
                new_node = RgNode(
                    self.petri_net_to_solve,
                    self.heuristic_function,
                    copy.deepcopy(self.rcpsp_base),
                    current,
                    transition,
                )
                if new_node not in available:
                    heapq.heappush(available, new_node)
            try:
                current = heapq.heappop(available)
                visited += 1
                if visited % 1000 == 0:
                    print(current.g_score, current.h_score)
                progress_bar.update(1)
            except IndexError:
                print(current.finished_activities)
                print(current.started_activities)
                print("No solution found")
        print(current.finished_activities)
        print(current.started_activities)
