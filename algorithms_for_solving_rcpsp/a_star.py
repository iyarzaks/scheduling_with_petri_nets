import copy
import heapq

from tqdm import tqdm

from RCPSP_modeling.rcpsp_petri_net import (
    PetriNetTransition,
    TIME,
    COUNT,
    FINISH,
    START,
)
from RCPSP_modeling.rcpsp_petri_net import RcpspTimedPlacePetriNet


class RgNode:
    def __init__(
        self,
        petri_net_to_solve,
        heuristic_function,
        rcpsp_base,
        previous_rg_node=None,
        transition: PetriNetTransition = None,
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
        if len(self.finished_activities) > 0:
            self.g_score = self.calc_g_score()
            # self.g_score = max(
            #     [
            #         rcpsp_base.find_activity_by_name(act).duration
            #         + self.started_activities[act]
            #         for act in self.started_activities
            #     ]
            # )
        else:
            self.g_score = 0
        self.available_transitions = self.check_available_transitions(
            petri_net_to_solve, current_time=None
        )

        self.h_score = heuristic_function(
            rcpsp_base,
            self.finished_activities,
            self.started_activities,
            current_time=self.g_score,
        )
        self.f_score = self.g_score + self.h_score

    def check_available_transitions(self, petri_net_to_solve, current_time=None):
        return [
            (transition, current_time)
            for transition in petri_net_to_solve.transitions
            if transition.is_available(self.marking)
        ]

    def calc_g_score(self):
        return max(self.finished_activities.values())

    def update_marking_and_activities(
        self,
        previous_marking,
        previous_finished_activities,
        previous_started_activities,
        transition: PetriNetTransition,
        petri_net_to_solve: RcpspTimedPlacePetriNet,
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
        return (
            self.f_score < other.f_score
            or (self.f_score == other.f_score and self.g_score > other.g_score)
            or (
                self.f_score == other.f_score
                and self.g_score == other.g_score
                and len(self.started_activities) > len(other.started_activities)
            )
            or (
                self.f_score == other.f_score
                and self.g_score == other.g_score
                and len(self.finished_activities) > len(other.finished_activities)
            )
        )

    def __eq__(self, other):
        return self.marking == other.marking


class RgNodeTimedTransition:
    def __init__(
        self,
        petri_net_to_solve,
        heuristic_function,
        rcpsp_base,
        previous_rg_node=None,
        transition=None,
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
                transition[0],
                petri_net_to_solve,
                firing_time=transition[1],
            )
        if len(self.finished_activities) > 0:
            self.g_score = self.calc_g_score()
            # self.g_score = max(
            #     [
            #         rcpsp_base.find_activity_by_name(act).duration
            #         + self.started_activities[act]
            #         for act in self.started_activities
            #     ]
            # )
        else:
            self.g_score = 0
        self.available_transitions = self.check_available_transitions(
            petri_net_to_solve, current_time=self.g_score
        )

        self.h_score = heuristic_function(
            rcpsp_base,
            self.finished_activities,
            self.started_activities,
            current_time=self.g_score,
        )
        self.f_score = self.g_score + self.h_score

    def check_available_transitions(self, petri_net_to_solve, current_time=None):
        available_transitions = []
        for transition in petri_net_to_solve.transitions:
            min_time = transition.is_available(self.marking, current_time=current_time)
            if min_time is not False:
                available_transitions.append((transition, min_time))
        return available_transitions

    def __lt__(self, other):
        return (
            self.f_score < other.f_score
            or (self.f_score == other.f_score and self.g_score > other.g_score)
            or (
                self.f_score == other.f_score
                and self.g_score == other.g_score
                and len(self.started_activities) > len(other.started_activities)
            )
            or (
                self.f_score == other.f_score
                and self.g_score == other.g_score
                and len(self.finished_activities) > len(other.finished_activities)
            )
        )

    def __eq__(self, other):
        return (
            self.f_score == other.f_score
            and self.finished_activities == other.finished_activities
            and self.started_activities == other.started_activities
        )

    def calc_g_score(self):
        finished_jobs_current_times = [
            v
            for v in self.finished_activities.values()
            if v <= max(self.started_activities.values())
        ]
        finished_jobs_current_times.append(0)
        return max(finished_jobs_current_times)

    @staticmethod
    def consume_resource(resource, amount, current_time):
        # Sort resource list by time in descending order
        if amount < 1:
            return resource
        resource = sorted(resource, key=lambda x: x.get(TIME, 0), reverse=True)

        for item in resource:
            if len(item) > 0:
                if item[TIME] <= current_time:
                    if amount <= item[COUNT]:
                        item[COUNT] -= amount
                        break
                    else:
                        amount -= item[COUNT]
                        item[COUNT] = 0

        # Remove items with count 0
        resource = [item for item in resource if item.get(COUNT, 0) > 0]

        return resource

    @staticmethod
    def return_resource(resource, amount, return_time):
        # Check if there is already an entry with the return_time
        if amount < 1:
            return resource
        found = False
        for item in resource:
            if len(item) > 0:
                if item["time"] == return_time:
                    item["count"] += amount
                    found = True
                    break

        # If no entry with the return_time was found, create a new one
        if not found:
            resource.append({"count": amount, "time": return_time})

        return resource

    def update_marking_and_activities(
        self,
        previous_marking,
        previous_finished_activities,
        previous_started_activities,
        transition: PetriNetTransition,
        petri_net_to_solve: RcpspTimedPlacePetriNet,
        firing_time=0,
    ):
        new_marking = previous_marking.copy()
        new_finished_activities = previous_finished_activities.copy()
        new_started_activities = previous_started_activities.copy()
        for in_node in transition.arcs_in:
            new_marking[in_node] = RgNodeTimedTransition.consume_resource(
                new_marking[in_node].copy(), transition.arcs_in[in_node], firing_time
            )
        for out_node in transition.arcs_out:
            new_marking[out_node] = RgNodeTimedTransition.return_resource(
                new_marking[out_node].copy(),
                transition.arcs_out[out_node],
                firing_time + transition.duration,
            )
        new_finished_activities[transition.name] = (
            firing_time + petri_net_to_solve.transitions_dict[transition.name].duration
        )
        new_started_activities[transition.name] = firing_time
        self.finished_activities = new_finished_activities
        self.started_activities = new_started_activities
        self.marking = new_marking


class AStarSolver:
    def __init__(
        self,
        petri_net_to_solve: RcpspTimedPlacePetriNet,
        rcpsp_base,
        heuristic_function,
        timed_transition=False,
    ):
        self.petri_net_to_solve = petri_net_to_solve
        self.rcpsp_base = rcpsp_base
        self.heuristic_function = heuristic_function
        if timed_transition:
            self.start_node = RgNodeTimedTransition(
                petri_net_to_solve, heuristic_function, rcpsp_base
            )
        else:
            self.start_node = RgNode(petri_net_to_solve, heuristic_function, rcpsp_base)
        self.timed_transition = timed_transition

    def is_final_node(self, node: RgNode) -> bool:
        final_places = [
            place
            for place in self.petri_net_to_solve.places
            if len(place.arcs_out) == 0
        ]
        if self.timed_transition:
            if all(
                node.marking[place.name][-1].get("count", 0) > 0
                for place in final_places
            ):
                return True
        else:
            if all(
                node.marking[place.name].get("count", 0) > 0 for place in final_places
            ):
                return True
        return False

    def solve(self, beam_search_size=None, logging=False):
        if logging:
            progress_bar = tqdm(total=1000000, desc="Processing")
        current = self.start_node
        available = []
        closed = []
        while not self.is_final_node(current):
            available_transitions = current.available_transitions
            for transition in available_transitions:
                if self.timed_transition:
                    new_node = RgNodeTimedTransition(
                        self.petri_net_to_solve,
                        self.heuristic_function,
                        copy.deepcopy(self.rcpsp_base),
                        copy.deepcopy(current),
                        transition,
                    )
                else:
                    new_node = RgNode(
                        self.petri_net_to_solve,
                        self.heuristic_function,
                        copy.deepcopy(self.rcpsp_base),
                        current,
                        transition[0],
                    )
                if new_node not in available and new_node not in closed:
                    heapq.heappush(available, new_node)
            try:
                current = heapq.heappop(available)
                if beam_search_size is not None:
                    heapq.heapify(available)
                    available = heapq.nsmallest(beam_search_size, available)
                closed.append(current)
                if logging:
                    if len(closed) % 100 == 0:
                        print(f"activities started: {current.started_activities}")
                        print(f"g score: {current.g_score}, h score: {current.h_score}")
                    progress_bar.update(1)
            except IndexError:
                print(current.started_activities)
                print("No solution found")
        # print(f"activities start times: {current.started_activities}")
        # print(f"makespan: {current.g_score}")
        return {
            "scheduling": current.started_activities,
            "makespan": current.g_score,
            "nodes_visited": len(closed),
            "solved": True,
            "beam_search_size": beam_search_size,
        }
