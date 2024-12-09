from collections import defaultdict
from itertools import product

from RCPSP_modeling.rcpsp_base import RcpspBase

# import pygraphviz as pgv

START = "_start"
FINISH = "_finish"
PRE = "_pre_"
POST = "post_"
PLACE = "place"
COUNT = "count"
TIME = "time"


def merge_dicts(dict1, dict2, dict3=None):
    dict2.update(dict1)
    if dict3:
        dict3.update(dict2)
        return dict3
    return dict2


def get_in_place_if_existed(place, list_to_search):
    if len(list_to_search) > 0:
        return list_to_search[place]
    else:
        return 0


class PetriNetTransition:
    def __init__(self, name, arcs_in, arcs_out, duration=None):
        self.name = name
        self.arcs_in = arcs_in
        self.arcs_out = arcs_out
        self.duration = duration

    @staticmethod
    def min_time_to_fulfill_demand(resource, demand):
        # Sort resource list by time in ascending order
        if demand == 0:
            return 0
        if len(resource) == 0 or len(resource[-1]) == 0:
            return None
        resource = [r for r in resource if len(r) > 0]
        resource = sorted(resource, key=lambda x: x[1])

        accumulated_count = 0
        min_time = None

        for item in resource:
            accumulated_count += item[0]
            min_time = item[1]
            if accumulated_count >= demand:
                break

        if accumulated_count < demand:
            return None  # Not enough resources to fulfill the demand

        return min_time

    def is_available(self, marking):

        max_min = 0
        for place in self.arcs_in:
            min_time_available = PetriNetTransition.min_time_to_fulfill_demand(
                resource=marking.get(place, []), demand=self.arcs_in[place]
            )
            if min_time_available is None:
                return False
            if min_time_available > max_min:
                max_min = min_time_available
            # min_times_available.add(min_time_available)
        # else:
        #     return max(min_times_available)
        return max_min


class PetriNetPlace:
    def __init__(self, name, arcs_in, arcs_out, duration=0, state=None):
        self.name = name
        self.arcs_in = arcs_in
        self.arcs_out = arcs_out
        if state is not None:
            self.state = state
        self.duration = duration


class RcpspTimedPetriNet:
    def __init__(self):
        self.transitions_dict = {}
        self.places_dict = {}
        # self.net = pgv.AGraph(directed=True)
        self.transitions = []
        self.alternatives = None
        self.places = []

    def update_net(self):
        for place in self.places:
            # self.net.add_node(
            #     place.name,
            #     shape="circle",
            #     color="lightblue",
            #     label=f"{place.name}\n",
            # )
            self.places_dict[place.name] = place
            # for successor in place.arcs_in:
            #     self.net.add_edge(successor, place.name, label=place.arcs_in[successor])
        for transition in self.transitions:
            # self.net.add_node(transition.name, shape="box", color="lightgreen")
            # for successor in transition.arcs_in:
            #     self.net.add_edge(
            #         successor, transition.name, label=transition.arcs_in[successor]
            #     )
            self.transitions_dict[transition.name] = transition

    # def plot(self, filename):
    #     self.net.layout(prog="dot")
    #     self.net.draw(filename)


class RcpspTimedTransitionPetriNet(RcpspTimedPetriNet):
    def update_places(self):
        pass

    def __init__(self, rcpsp_basic: RcpspBase):

        # for nuw we assume that the activities sorted in a way that if for every i,
        # j if i<j j nod depend on i in any case

        super().__init__()
        self.update_places()
        for resource in rcpsp_basic.resources:
            self.places.append(
                PetriNetPlace(
                    name=resource,
                    arcs_in={
                        activity.name: activity.resource_demands[resource]
                        for activity in rcpsp_basic.activities
                        if resource in activity.resource_demands.keys()
                    },
                    arcs_out={
                        activity.name: activity.resource_demands[resource]
                        for activity in rcpsp_basic.activities
                        if resource in activity.resource_demands.keys()
                    },
                    state=[[rcpsp_basic.resources[resource], 0]],
                )
            )
        for activity in rcpsp_basic.activities:
            self.add_activity(activity, rcpsp_basic)
        self.update_net()

    def add_activity(self, activity, rcpsp_basic):
        if activity.name not in rcpsp_basic.backward_dependencies:
            # no dependent activity add start place
            self.places.append(
                PetriNetPlace(
                    name=PRE + activity.name,
                    arcs_in=dict(),
                    arcs_out={activity.name: 1},
                    duration=0,
                    state=[[1, 0]],
                )
            )
        post_no_dependencies_dict = (
            {POST + activity.name: 1}
            if activity.name not in rcpsp_basic.dependencies
            or len(rcpsp_basic.dependencies[activity.name]) == 0
            else {}
        )
        self.transitions.append(
            # add activity transition
            PetriNetTransition(
                name=activity.name,
                arcs_in=merge_dicts(
                    {
                        place.name: 1
                        for place in self.places
                        if activity.name in place.arcs_out
                        and place.name not in rcpsp_basic.resources
                    },
                    {
                        resource: activity.resource_demands[resource]
                        for resource in activity.resource_demands.keys()
                    },
                ),
                duration=activity.duration,
                arcs_out=merge_dicts(
                    {
                        POST + activity.name + PRE + activity_successor: 1
                        for activity_successor in rcpsp_basic.dependencies.get(
                            activity.name, []
                        )
                    },
                    post_no_dependencies_dict,
                    {
                        resource: activity.resource_demands[resource]
                        for resource in activity.resource_demands.keys()
                    },
                ),
            )
        )

        for successor_activity in rcpsp_basic.dependencies.get(activity.name, []):
            self.places.append(
                PetriNetPlace(
                    name=POST + activity.name + PRE + successor_activity,
                    arcs_in={activity.name: 1},
                    arcs_out={successor_activity: 1},
                )
            )
        if (
            activity.name not in rcpsp_basic.dependencies
            or len(rcpsp_basic.dependencies[activity.name]) == 0
        ):
            self.places.append(
                PetriNetPlace(
                    name=POST + activity.name,
                    arcs_in={activity.name: 1},
                    arcs_out=dict(),
                )
            )


class RcpspTimedTransitionPetriNet_As(RcpspTimedTransitionPetriNet):

    def update_places(self):
        self.places = (
            self.places
            + list(self.special_places_xor_join.values())
            + list(self.special_places_xor_split.values())
        )
        # to_remove = []
        # for place in self.places:
        #     if len(place.arcs_out.keys()) == 1:
        #         arc_out_node = list(place.arcs_out.keys())[0]
        #         if (
        #             arc_out_node
        #             in self.gather_activities_with_same_successors_diff_branches
        #         ):
        #             to_remove.append(place)
        # for place in to_remove:
        #     self.places.remove(place)

    @staticmethod
    def get_job_combinations(group_lists, job_dict):
        first_list, second_list = group_lists
        combinations = list(product(first_list, second_list))

        result = {}

        for combo in combinations:
            jobs_in_combo = [
                job
                for job, groups in job_dict.items()
                if any(group in combo for group in groups["branch_ids"])
            ]
            result[combo] = jobs_in_combo

        return result

    def __init__(self, rcpsp_basic: RcpspBase, activities_branches_list, subgraphs):
        self.activities_branches_list = activities_branches_list
        self.subgraphs = subgraphs

        self.splitted_branch = {}
        self.gather_activities_with_same_successors_diff_branches = defaultdict(list)
        self.gather_activities_with_same_predecessor_diff_branches = defaultdict(list)
        (
            self.special_places_xor_join,
            self.special_places_xor_split,
            self.splitted_branch,
        ) = self.add_special_places_and_transitions(rcpsp_basic)

        super().__init__(rcpsp_basic)
        self.alternatives = RcpspTimedTransitionPetriNet_As.get_job_combinations(
            [sg["branch_ids"] for sg in subgraphs], self.activities_branches_list
        )
        main_branch_jobs = [
            job
            for job, groups in self.activities_branches_list.items()
            if groups["branch_ids"] == [1]
        ]
        for alt in self.alternatives:
            self.alternatives[alt] += main_branch_jobs
        for alt in self.alternatives:
            not_relevant = list(
                set(rcpsp_basic.activities_names_durations)
                - set(self.alternatives[alt])
            )
            self.alternatives[alt] = not_relevant

    def add_special_places_and_transitions(self, rcpsp_basic):
        special_places_xor_join = {}
        special_places_xor_split = {}
        branch_splits = {}
        for dependency in rcpsp_basic.backward_dependencies.items():
            if len(dependency[1]) > 1 and (
                1 in self.activities_branches_list[dependency[0]]["branch_ids"]
                or len(self.activities_branches_list[dependency[0]]["branch_ids"]) > 1
            ):
                for activity in dependency[1]:
                    if 1 not in self.activities_branches_list[activity]["branch_ids"]:
                        self.gather_activities_with_same_successors_diff_branches[
                            dependency[0]
                        ].append(activity)

        for dependency in rcpsp_basic.dependencies.items():
            if (
                len(dependency[1]) > 1
                and 1 in self.activities_branches_list[dependency[0]]["branch_ids"]
            ):
                for activity in dependency[1]:
                    if 1 not in self.activities_branches_list[activity]["branch_ids"]:
                        self.gather_activities_with_same_predecessor_diff_branches[
                            dependency[0]
                        ].append(activity)
            if (
                len(dependency[1]) > 1
                and 1 not in self.activities_branches_list[dependency[0]]["branch_ids"]
                and len(
                    set(
                        [
                            tuple(self.activities_branches_list[act]["branch_ids"])
                            for act in dependency[1]
                        ]
                    )
                )
                > 1
            ):
                all_succesors_branches = [
                    self.activities_branches_list[depend]["branch_ids"]
                    for depend in dependency[1]
                ]
                unique_union = {
                    item for sublist in all_succesors_branches for item in sublist
                }
                unique_union = list(
                    unique_union
                    - {self.activities_branches_list[dependency[0]]["branch_ids"][0]}
                )
                branch_splits[
                    self.activities_branches_list[dependency[0]]["branch_ids"][0]
                ] = unique_union

        for (
            special_case
        ) in self.gather_activities_with_same_successors_diff_branches.items():
            post_str = ""
            for activity_name in special_case[1]:
                post_str += POST + activity_name
            post_str += PRE + special_case[0]

            special_places_xor_join[special_case[0]] = PetriNetPlace(
                name=post_str,
                arcs_in={a: 1 for a in special_case[1]},
                arcs_out={special_case[0]: 1},
                duration=0,
            )

        for (
            special_case
        ) in self.gather_activities_with_same_predecessor_diff_branches.items():
            pre_str = ""
            for activity_name in special_case[1]:
                pre_str += PRE + activity_name
            pre_str = POST + special_case[0] + pre_str

            special_places_xor_split[special_case[0]] = PetriNetPlace(
                name=pre_str,
                arcs_in={special_case[0]: 1},
                arcs_out={a: 1 for a in special_case[1]},
                duration=0,
            )

        for branch_split in branch_splits:
            # last_in_splitted =
            # last_in_split_to = []
            # add place which gather all lasts
            # remove arc_out from last splitted
            # add transitions form place to relevant "before main graph" place
            # in_activities = [last_act_in()]
            # last_act_in_splitted =

            special_places_xor_split[special_case[0]] = PetriNetPlace(
                name=pre_str,
                arcs_in={special_case[0]: 1},
                arcs_out={a: 1 for a in special_case[1]},
                duration=0,
            )

        return special_places_xor_join, special_places_xor_split, branch_split

    def add_activity(self, activity, rcpsp_basic):
        if activity.name not in rcpsp_basic.backward_dependencies:
            # no dependent activity add start place
            self.places.append(
                PetriNetPlace(
                    name=PRE + activity.name,
                    arcs_in=dict(),
                    arcs_out={activity.name: 1},
                    duration=0,
                    state=[[1, 0]],
                )
            )
        post_no_dependencies_dict = (
            {POST + activity.name: 1}
            if activity.name not in rcpsp_basic.dependencies
            or len(rcpsp_basic.dependencies[activity.name]) == 0
            else {}
        )
        arcs_out = self.extract_arcs_out(
            activity, post_no_dependencies_dict, rcpsp_basic
        )
        self.transitions.append(
            # add activity transition
            PetriNetTransition(
                name=activity.name,
                arcs_in=self.extract_arcs_in(activity, rcpsp_basic),
                duration=activity.duration,
                arcs_out=arcs_out,
            )
        )

        for successor_activity in rcpsp_basic.dependencies.get(activity.name, []):
            if (
                successor_activity
                in self.gather_activities_with_same_successors_diff_branches
                or (
                    activity.name
                    in self.gather_activities_with_same_predecessor_diff_branches
                    and successor_activity
                    in self.gather_activities_with_same_predecessor_diff_branches[
                        activity.name
                    ]
                )
            ):
                continue
            else:
                self.places.append(
                    PetriNetPlace(
                        name=POST + activity.name + PRE + successor_activity,
                        arcs_in={activity.name: 1},
                        arcs_out={successor_activity: 1},
                    )
                )
        if (
            activity.name not in rcpsp_basic.dependencies
            or len(rcpsp_basic.dependencies[activity.name]) == 0
        ):
            self.places.append(
                PetriNetPlace(
                    name=POST + activity.name,
                    arcs_in={activity.name: 1},
                    arcs_out=dict(),
                )
            )

    def extract_arcs_out(self, activity, post_no_dependencies_dict, rcpsp_basic):
        dependents_dict = {}
        for activity_successor in rcpsp_basic.dependencies.get(activity.name, []):
            if (
                activity_successor
                in self.gather_activities_with_same_successors_diff_branches
            ):
                dependents_dict[
                    self.special_places_xor_join[activity_successor].name
                ] = 1
            elif (
                activity.name
                in self.gather_activities_with_same_predecessor_diff_branches
            ):
                dependents_dict[self.special_places_xor_split[activity.name].name] = 1
            else:
                dependents_dict[POST + activity.name + PRE + activity_successor] = 1
        return merge_dicts(
            dependents_dict,
            post_no_dependencies_dict,
            {
                resource: activity.resource_demands[resource]
                for resource in activity.resource_demands.keys()
            },
        )

    def extract_arcs_in(self, activity, rcpsp_basic):
        dependents_dict = {}
        for activity_successor in rcpsp_basic.dependencies.get(activity.name, []):
            if (
                activity_successor
                in self.gather_activities_with_same_successors_diff_branches
            ):
                dependents_dict[
                    self.special_places_xor_join[activity_successor].name
                ] = 1
            elif (
                activity.name
                in self.gather_activities_with_same_predecessor_diff_branches
                and activity_successor
                in self.gather_activities_with_same_predecessor_diff_branches[
                    activity.name
                ]
            ):
                dependents_dict[self.special_places_xor_split[activity.name].name] = 1
            else:
                dependents_dict[POST + activity.name + PRE + activity_successor] = 1
        return merge_dicts(
            {
                place.name: 1
                for place in self.places
                if activity.name in place.arcs_out
                and place.name not in rcpsp_basic.resources
            },
            {
                resource: activity.resource_demands[resource]
                for resource in activity.resource_demands.keys()
            },
        )


class RcpspTimedPlacePetriNet(RcpspTimedPetriNet):
    def __init__(self, rcpsp_basic: RcpspBase):

        # for nuw we assume that the activities sorted in a way that if for every i,
        # j if i<j j nod depend on i in any case
        super().__init__()
        for resource in rcpsp_basic.resources:
            self.places.append(
                PetriNetPlace(
                    name=resource,
                    arcs_in={
                        activity.name + FINISH: activity.resource_demands[resource]
                        for activity in rcpsp_basic.activities
                        if resource in activity.resource_demands.keys()
                    },
                    arcs_out={
                        activity.name + START: activity.resource_demands[resource]
                        for activity in rcpsp_basic.activities
                        if resource in activity.resource_demands.keys()
                    },
                    state=[rcpsp_basic.resources[resource], 0],
                )
            )
        for activity in rcpsp_basic.activities:
            self.add_activity(activity, rcpsp_basic)
        self.update_net()

    def add_activity(self, activity, rcpsp_basic):
        if activity.name not in rcpsp_basic.backward_dependencies:
            # no dependent activity add start place
            self.places.append(
                PetriNetPlace(
                    name=PRE + activity.name,
                    arcs_in=dict(),
                    arcs_out={activity.name + START: 1},
                    duration=0,
                    state=[1, 0],
                )
            )
        self.places.append(
            # add activity place
            PetriNetPlace(
                name=activity.name,
                arcs_in={
                    activity.name + START: sum(activity.resource_demands.values()) + 1
                },
                arcs_out={
                    activity.name + FINISH: sum(activity.resource_demands.values()) + 1
                },
                duration=activity.duration,
            )
        )
        self.transitions.append(
            # add start of activity transition
            PetriNetTransition(
                name=activity.name + START,
                arcs_in=merge_dicts(
                    {
                        place.name: 1
                        for place in self.places
                        if activity.name + START in place.arcs_out
                        and place.name not in rcpsp_basic.resources
                    },
                    {
                        resource: activity.resource_demands[resource]
                        for resource in activity.resource_demands.keys()
                    },
                ),
                arcs_out={activity.name: sum(activity.resource_demands.values()) + 1},
            )
        )
        post_no_dependencies_dict = (
            {POST + activity.name: 1}
            if activity.name not in rcpsp_basic.dependencies
            or len(rcpsp_basic.dependencies[activity.name]) == 0
            else {}
        )
        self.transitions.append(
            # add end of activity transition
            PetriNetTransition(
                name=activity.name + FINISH,
                arcs_in={activity.name: sum(activity.resource_demands.values()) + 1},
                arcs_out=merge_dicts(
                    {
                        POST + activity.name + PRE + activity_successor: 1
                        for activity_successor in rcpsp_basic.dependencies.get(
                            activity.name, []
                        )
                    },
                    post_no_dependencies_dict,
                    {
                        resource: activity.resource_demands[resource]
                        for resource in activity.resource_demands.keys()
                    },
                ),
            )
        )
        for successor_activity in rcpsp_basic.dependencies.get(activity.name, []):
            self.places.append(
                PetriNetPlace(
                    name=POST + activity.name + PRE + successor_activity,
                    arcs_in={activity.name + FINISH: 1},
                    arcs_out={successor_activity + START: 1},
                )
            )
        if (
            activity.name not in rcpsp_basic.dependencies
            or len(rcpsp_basic.dependencies[activity.name]) == 0
        ):
            self.places.append(
                PetriNetPlace(
                    name=POST + activity.name,
                    arcs_in={activity.name + FINISH: 1},
                    arcs_out=dict(),
                )
            )
