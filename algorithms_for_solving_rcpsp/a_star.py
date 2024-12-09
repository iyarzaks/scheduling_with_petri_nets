import copy
import hashlib
import heapq
from types import MappingProxyType

import orjson
from tqdm import tqdm

from RCPSP_modeling.rcpsp_petri_net import (
    PetriNetTransition,
    TIME,
    COUNT,
    FINISH,
    START,
    get_in_place_if_existed,
    RcpspTimedPetriNet,
)
from RCPSP_modeling.rcpsp_petri_net import RcpspTimedPlacePetriNet


def selective_deep_copy(original):
    copy_dict = {}
    for key, value in original.items():
        if "R" in key:
            copy_dict[key] = value.copy()
        else:
            copy_dict[key] = value
    return copy_dict


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

    def update_marking_and_neighbors(self):
        pass


class RgNodeTimedTransition:
    def __init__(
        self,
        petri_net_to_solve,
        heuristic_function,
        rcpsp_base,
        previous_marking=None,
        previous_rg_node_finished_activities=None,
        previous_rg_node_started_activities=None,
        transition=None,
        job_finish_activity=None,
        heuristic_params=None,
    ):
        self.available_transitions = None
        self.previous_marking = previous_marking
        self.marking = {}
        self.petri_net_to_solve = petri_net_to_solve
        self.rcpsp_base = rcpsp_base
        self.transition = transition
        if previous_rg_node_finished_activities is None:
            self.finished_activities = {}
            self.started_activities = {}
            for place in petri_net_to_solve.places:
                if hasattr(place, "state"):
                    self.marking[place.name] = place.state
        else:
            self.firing_time = transition[1]
            self.update_activities(
                previous_finished_activities=previous_rg_node_finished_activities,
                previous_started_activities=previous_rg_node_started_activities,
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
        # self.available_transitions = self.check_available_transitions(
        #     petri_net_to_solve, current_time=self.g_score
        # )

        self.h_score = heuristic_function(
            rcpsp_base,
            self.finished_activities,
            self.started_activities,
            current_time=self.g_score,
            job_finish_activity=job_finish_activity,
            alternatives=self.petri_net_to_solve.alternatives,
            # heuristic_params=heuristic_params,
        )
        self.f_score = self.g_score + self.h_score

    def check_available_transitions(self, optional_transitions=None):
        available_transitions = []
        for transition in optional_transitions:
            min_time = transition.is_available(self.marking)
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
        return self.hash == other.hash

    @staticmethod
    def hash_dict(d):
        """
        Hash a dictionary using SHA-256.

        Parameters:
        d (dict): The dictionary to hash.

        Returns:
        int: The integer representation of the hash.
        """
        # Convert the dictionary to a JSON string with sorted keys using orjson
        dict_bytes = orjson.dumps(d, option=orjson.OPT_SORT_KEYS)

        # Create a SHA-256 hash object and update it with the bytes of the JSON string
        hash_obj = hashlib.sha256(dict_bytes)

        # Return the hexadecimal hash string as an integer
        return int(hash_obj.hexdigest(), 16)

    def __hash__(self):
        return self.hash

    def calc_g_score(self):
        return max(self.finished_activities.values())
        # finished_jobs_current_times = list(self.finished_activities.values())
        # # finished_jobs_current_times = [
        # #     v
        # #     for v in self.finished_activities.values()
        # #     if v <= max(self.started_activities.values())
        # # ]
        # finished_jobs_current_times.append(0)
        # return max(finished_jobs_current_times)

    @staticmethod
    def consume_resource(resource, amount, current_time):
        # Create a shallow copy of the resource list
        resource_copy = [item[:] for item in resource]

        # Sort resource list by time in descending order
        if amount < 1:
            return resource_copy
        resource_copy.sort(key=lambda x: get_in_place_if_existed(1, x), reverse=True)

        for item in resource_copy:
            if len(item) > 0:
                if item[1] <= current_time:
                    if amount <= item[0]:
                        item[0] -= amount
                        break
                    else:
                        amount -= item[0]
                        item[0] = 0

        # Remove items with count 0
        resource_copy = [
            item for item in resource_copy if get_in_place_if_existed(0, item) > 0
        ]

        return resource_copy

    @staticmethod
    def return_resource(resource, amount, return_time):
        # Create a shallow copy of the resource list
        resource_copy = [item[:] for item in resource]

        # Check if there is already an entry with the return_time
        if amount < 1:
            return resource_copy
        found = False
        for item in resource_copy:
            if len(item) > 0:
                if item[1] == return_time:
                    item[0] += amount
                    found = True
                    break

        # If no entry with the return_time was found, create a new one
        if not found:
            resource_copy.append([amount, return_time])

        return resource_copy

    def update_activities(
        self,
        previous_finished_activities,
        previous_started_activities,
    ):
        new_finished_activities = dict(previous_finished_activities)
        new_started_activities = dict(previous_started_activities)
        new_finished_activities[self.transition[0].name] = (
            self.firing_time + self.transition[0].duration
        )
        new_started_activities[self.transition[0].name] = self.firing_time
        self.finished_activities = new_finished_activities
        self.started_activities = new_started_activities
        self.hash = RgNodeTimedTransition.hash_dict(self.started_activities)

    def update_marking_and_activities(
        self,
        previous_marking,
        previous_finished_activities,
        previous_started_activities,
        transition: PetriNetTransition,
        petri_net_to_solve: RcpspTimedPlacePetriNet,
        firing_time=0,
    ):
        new_marking = previous_marking
        new_finished_activities = dict(previous_finished_activities)
        new_started_activities = dict(previous_started_activities)
        for in_node in transition.arcs_in:
            new_marking[in_node] = RgNodeTimedTransition.consume_resource(
                new_marking[in_node].copy(), transition.arcs_in[in_node], firing_time
            )
            if not new_marking[in_node]:
                del new_marking[in_node]
        for out_node in transition.arcs_out:
            new_marking[out_node] = RgNodeTimedTransition.return_resource(
                new_marking.get(out_node, [[]]).copy(),
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

    @staticmethod
    def clean_empty_lists(data):
        """
        This function removes empty lists from a dictionary where each value is a list of lists.

        Parameters:
        data (dict): The dictionary to clean.

        Returns:
        dict: The cleaned dictionary with empty lists removed.
        """
        cleaned_data = {}
        for key, value in data.items():
            # Filter out empty lists from the value
            cleaned_value = [lst for lst in value if lst]
            # Add the cleaned value to the new dictionary
            cleaned_data[key] = cleaned_value
        return cleaned_data

    def update_marking_and_neighbors(self):
        if self.previous_marking is not None:
            new_marking = {k: v[:] for k, v in self.previous_marking.items()}

            for in_node in self.transition[0].arcs_in:
                new_marking[in_node] = RgNodeTimedTransition.consume_resource(
                    new_marking[in_node],
                    self.transition[0].arcs_in[in_node],
                    self.firing_time,
                )
                if not new_marking[in_node]:
                    del new_marking[in_node]
            for out_node in self.transition[0].arcs_out:
                new_marking[out_node] = RgNodeTimedTransition.return_resource(
                    new_marking.get(out_node, [[]]),
                    self.transition[0].arcs_out[out_node],
                    self.firing_time
                    + self.petri_net_to_solve.transitions_dict[
                        self.transition[0].name
                    ].duration,
                )
            self.marking = RgNodeTimedTransition.clean_empty_lists(new_marking)
        optional_transitions = self.get_optional_transitions()
        self.available_transitions = self.check_available_transitions(
            optional_transitions=optional_transitions
        )

    def get_optional_transitions(self):
        if not self.transition:
            return self.petri_net_to_solve.transitions

        # Cache repeated attribute accesses
        transitions_dict = self.petri_net_to_solve.transitions_dict
        started_activities = self.started_activities
        started_activities_keys = set(
            started_activities.keys()
        )  # Cache keys to a set for faster lookups

        optional_transitions = set()
        for transition in self.petri_net_to_solve.transitions:
            if transition.name not in started_activities:
                dependent_acts = set(
                    self.rcpsp_base.backward_dependencies[transition.name]
                )
                if len(dependent_acts.intersection(started_activities_keys)) > 0:
                    optional_transitions.add(transition)

        return optional_transitions


class AStarSolver:
    def __init__(
        self,
        petri_net_to_solve: RcpspTimedPetriNet,
        rcpsp_base,
        heuristic_function,
        timed_transition=False,
        job_finish_activity=None,
        heuristic_params=None,
    ):
        self.petri_net_to_solve = petri_net_to_solve
        self.rcpsp_base = rcpsp_base
        self.heuristic_function = heuristic_function
        if timed_transition:
            self.start_node = RgNodeTimedTransition(
                petri_net_to_solve,
                heuristic_function,
                rcpsp_base,
                job_finish_activity=job_finish_activity,
                heuristic_params=heuristic_params,
            )
        else:
            self.start_node = RgNode(petri_net_to_solve, heuristic_function, rcpsp_base)
        if job_finish_activity is not None:
            self.job_finish_activity = job_finish_activity

        self.timed_transition = timed_transition
        self.heuristic_params = heuristic_params

    def is_final_node(self, node: RgNode) -> bool:
        final_places = [
            place
            for place in self.petri_net_to_solve.places
            if len(place.arcs_out) == 0
        ]
        if self.timed_transition:
            if all(
                get_in_place_if_existed(0, node.marking.get(place.name, [[]])[-1]) > 0
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
        current.update_marking_and_neighbors()
        generated = 0
        available = []
        closed = set()
        while not self.is_final_node(current):
            available_transitions = current.available_transitions
            # if len(closed) % 1000 == 0:
            #     print("stop")
            for transition in available_transitions:
                if self.timed_transition:
                    generated += 1
                    new_node = RgNodeTimedTransition(
                        previous_marking=current.marking,
                        petri_net_to_solve=self.petri_net_to_solve,
                        heuristic_function=self.heuristic_function,
                        rcpsp_base=self.rcpsp_base,
                        previous_rg_node_finished_activities=MappingProxyType(
                            current.finished_activities
                        ),
                        previous_rg_node_started_activities=MappingProxyType(
                            current.started_activities
                        ),
                        transition=transition,
                        job_finish_activity=self.job_finish_activity,
                        heuristic_params=self.heuristic_params,
                    )
                else:
                    new_node = RgNode(
                        self.petri_net_to_solve,
                        self.heuristic_function,
                        copy.copy(self.rcpsp_base),
                        current,
                        transition[0],
                    )
                if new_node not in closed:
                    heapq.heappush(available, new_node)
            try:
                current = heapq.heappop(available)
                # print(current.h_score, current.g_score)
                if beam_search_size is not None:
                    heapq.heapify(available)
                    available = heapq.nsmallest(beam_search_size, available)
                closed.add(current)
                current.update_marking_and_neighbors()
                if logging:
                    progress_bar.update(1)
            except IndexError:
                print(current.started_activities)
                print("No solution found")
                return {
                    "scheduling": None,
                    "makespan": None,
                    "nodes_visited": len(closed),
                    "solved": False,
                    "beam_search_size": beam_search_size,
                }
        # print(f"activities start times: {current.started_activities}")
        # print(f"makespan: {current.g_score}")
        return {
            "scheduling": current.started_activities,
            "total_jobs_scheduled": len(current.started_activities),
            "makespan": current.started_activities[self.job_finish_activity],
            "nodes_expand": len(closed),
            "nodes_generated": generated,
            "solved": True,
            "beam_search_size": beam_search_size,
        }
