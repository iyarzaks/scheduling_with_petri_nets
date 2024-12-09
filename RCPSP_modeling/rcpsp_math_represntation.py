from typing import Dict, Any

import numpy as np
import pandas as pd

from RCPSP_modeling.rcpsp_base import RcpspBase
from algorithms_for_solving_rcpsp.solve_with_mip_solver import (
    solve_rcpsp_lp_relaxation_optimized,
)
from extract_problems.extract_problem import extract_rcpsp, extract_rcpsp_for_solver


def safe_max(values):
    """
    Return the maximum value in the list, or 0 if the list is empty.

    Args:
    values (list): A list of numerical values.

    Returns:
    float or int: The maximum value in the list, or 0 if the list is empty.
    """
    return max(values) if values else 0


def from_rcpsp_to_incidence_matrix(rcpsp_example):
    transitions = []
    for activity in rcpsp_example.activities:
        transitions.append(f"start_{activity.name}")
        transitions.append(f"end_{activity.name}")

    # Create an empty DataFrame for the matrix
    places = (
        [f"{activity.name}" for activity in rcpsp_example.activities]
        + [f"resource_{resource}" for resource in rcpsp_example.resources.keys()]
        + [
            f"dep_{key}->{value}"
            for key, values in rcpsp_example.dependencies.items()
            for value in values
        ]
        + ["start", "finish"]
    )

    matrix = pd.DataFrame(0, index=places, columns=transitions)

    # Fill the matrix with values
    for activity in rcpsp_example.activities:
        start_event = f"start_{activity.name}"
        end_event = f"end_{activity.name}"

        # Update for activity start
        matrix.at[activity.name, start_event] = 1
        for resource, demand in activity.resource_demands.items():
            matrix.at[f"resource_{resource}", start_event] = -demand

        # Update for activity end
        matrix.at[activity.name, end_event] = -1
        for resource, demand in activity.resource_demands.items():
            matrix.at[f"resource_{resource}", end_event] = demand

        # Handle dependencies
        if activity.name in rcpsp_example.dependencies:
            for dependent_activity in rcpsp_example.dependencies[activity.name]:
                matrix.at[f"dep_{activity.name}->{dependent_activity}", end_event] = 1
                matrix.at[
                    f"dep_{activity.name}->{dependent_activity}",
                    f"start_{dependent_activity}",
                ] = -1
    matrix.at["start", "start_1"] = -1
    matrix.at["finish", f"end_{rcpsp_example.activities[-1].name}"] = 1
    return matrix


def init_real_problem(file_path):
    rcpsp_example = extract_rcpsp(file_path)
    matrix = from_rcpsp_to_incidence_matrix(rcpsp_example)
    return matrix, rcpsp_example


def build_cp_heuristic(rcpsp):
    def cp_heuristic(started_activities, current_time, finished_activities):
        ongoing = {}
        if started_activities:
            for act, start_time in started_activities.items():
                finish_time = rcpsp.activities_names_durations[act] + start_time
                if act not in finished_activities:
                    ongoing[act] = max(finish_time - current_time, 0)
        return rcpsp.get_all_critical_path_of_sub(
            list(finished_activities), rcpsp.activities[-1].name, ongoing
        )[0]

    return cp_heuristic


def build_lp_heuristic(path):
    p, u, e, c = extract_rcpsp_for_solver(path)

    def lp_heuristic(started_activities, current_time):

        res = solve_rcpsp_lp_relaxation_optimized(p, u, e, c, started_activities)
        return res

    return lp_heuristic


def is_dict_contained(small_dict, big_dict):
    return all(item in big_dict.items() for item in small_dict.items())


def fastest_hash_two_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> int:
    """
    Fastest implementation for numeric/string values, sacrificing some collision resistance.
    """
    hash_value = 0
    for k, v in sorted(d1.items()):
        hash_value ^= hash((k, v))
    for k, v in sorted(d2.items()):
        hash_value ^= hash((k, v))
    return hash_value


class AstarNode:
    def __init__(
        self,
        places_vector,
        heuristic_function,
        started_activities,
        finished_activities,
    ):

        self.g_score = safe_max(finished_activities.values())
        self.g_score = safe_max(finished_activities.values())

        self.places_vector = places_vector
        self.started_activities = started_activities
        self.h_score = heuristic_function(
            started_activities, self.g_score, finished_activities
        )
        self.f_score = self.g_score + self.h_score
        self.finished_activities = finished_activities

    def __hash__(self):
        return fastest_hash_two_dicts(self.started_activities, self.finished_activities)

    def __lt__(self, other):
        return (
            self.f_score < other.f_score
            or (self.f_score == other.f_score and self.g_score > other.g_score)
            or (
                self.f_score == other.f_score
                and self.g_score == other.g_score
                and len(self.finished_activities) > len(other.finished_activities)
            )
            or (
                self.f_score == other.f_score
                and self.g_score == other.g_score
                and len(self.started_activities) > len(other.started_activities)
            )
        )

    def __eq__(self, other):
        return (
            # np.array_equal(self.places_vector, other.places_vector)
            self.started_activities == other.started_activities
            and self.finished_activities == other.finished_activities
        )


class AstarOptimizedSolver:
    def __init__(self, matrix, rcpsp, heuristic_function):
        self.heuristic_function = heuristic_function
        self.matrix = matrix
        self.matrix_values = self.matrix.values
        self.rcpsp = rcpsp
        self.started_activities = {}

    def solve_a_star(self, logging=False):
        init_marking, final_marking = self.get_init_marking_and_final_markings()
        start_node = AstarNode(
            places_vector=init_marking,
            heuristic_function=self.heuristic_function,
            started_activities={},
            finished_activities={},
        )

        open_set = []
        heapq.heappush(open_set, start_node)
        open_set_hash = {start_node}
        closed_set = set()
        generated = 0

        if logging:
            progress_bar = tqdm(desc="Processing")

        while open_set:
            current = heapq.heappop(open_set)
            # print(current.started_activities)
            # print(current.g_score, current.h_score, current.f_score)
            # print(current.h_score, current.g_score)

            # open_set_hash.remove(current)

            if np.array_equal(current.places_vector, final_marking):
                if logging:
                    progress_bar.close()
                return self._get_result_dict(current, len(closed_set), generated)

            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                generated += 1
                if neighbor in closed_set:
                    continue
                if neighbor.f_score < current.f_score:
                    print("Not consistent")
                heapq.heappush(open_set, neighbor)
                open_set_hash.add(neighbor)

            if logging:
                progress_bar.update(1)
                if len(closed_set) % 10000 == 0:
                    print(f"g score: {current.g_score}, h score: {current.h_score}")
                    print(f"finished activities: {current.finished_activities}")

        if logging:
            progress_bar.close()
        return {"solved": False}

    def _get_result_dict(self, final_node, nodes_expanded, nodes_generated):
        return {
            "scheduling": final_node.started_activities,
            "total_jobs_scheduled": len(final_node.started_activities),
            "makespan": final_node.started_activities[self.rcpsp.activities[-1].name],
            "nodes_expanded": nodes_expanded,
            "nodes_generated": nodes_generated,
            "solved": True,
        }

    def get_init_marking_and_final_markings(self):
        init_marking = [0] * (self.matrix.shape[0] - 2) + [1, 0]
        resources_rows = self.matrix.index.str.contains("resource")
        matching_indices = self.matrix.index[resources_rows].tolist()
        numerical_indices = [self.matrix.index.get_loc(idx) for idx in matching_indices]
        for resource, idx in zip(matching_indices, numerical_indices):
            init_marking[idx] = self.rcpsp.resources[resource.replace("resource_", "")]
        final_marking = init_marking.copy()
        final_marking[-2] = 0
        final_marking[-1] = 1
        init_marking = np.array(init_marking)
        final_marking = np.array(final_marking)
        return init_marking, final_marking

    def get_neighbors_bb(self, current):
        indices_of_available_transitions, new_markings = self.get_available_indices(
            current
        )
        neighbours = []

        for i in range(len(indices_of_available_transitions)):
            idx = indices_of_available_transitions[i]
            vector = new_markings[:, i]
            finished_activities_new, new_marking, started_activities_new = (
                self.get_new_neighbor_data(current, idx, vector)
            )
            # if finished_activities_new == {"1": 0, "2": 10}:
            #     print("A")

            # Ensure started_activities and finished_activities are properly updated
            new_started_activities = current.started_activities.copy()
            new_finished_activities = current.finished_activities.copy()

            # Update with new activities
            new_started_activities.update(started_activities_new)
            new_finished_activities.update(finished_activities_new)

            # Calculate the makespan (maximum end time of all started activities)
            makespan = safe_max(new_finished_activities.values())

            # Calculate the lower bound
            heuristic_value = self.heuristic_function(
                new_started_activities,
                safe_max(new_finished_activities.values()),
                new_finished_activities,
            )
            neighbours.append(
                BranchAndBoundNode(
                    places_vector=new_marking,
                    started_activities=new_started_activities,
                    finished_activities=new_finished_activities,
                    lower_bound=heuristic_value + makespan,
                )
            )

        return neighbours

    def get_neighbors(self, current):
        indices_of_available_transitions, new_markings = self.get_available_indices(
            current
        )
        neighbours = []
        for i in range(len(indices_of_available_transitions)):
            idx = indices_of_available_transitions[i]
            vector = new_markings[:, i]
            finished_activities_new, new_marking, started_activities_new = (
                self.get_new_neighbor_data(current, idx, vector)
            )
            neighbours.append(
                AstarNode(
                    new_marking,
                    heuristic_function=self.heuristic_function,
                    started_activities=started_activities_new,
                    finished_activities=finished_activities_new,
                )
            )
        return neighbours

    def get_new_neighbor_data(self, current, i, new_marking):
        transition = self.matrix.columns[i]
        finished_activities_new, started_activities_new = self.get_new_started_finished(
            current, transition
        )
        return finished_activities_new, new_marking, started_activities_new

    def get_new_started_finished(self, current, transition):
        finished_activities_new = current.finished_activities
        started_activities_new = current.started_activities
        if "start" in transition:
            started_activities_new = current.started_activities.copy()
            started_activities_new[transition.replace("start_", "")] = safe_max(
                current.finished_activities.values()
            )
        if "end" in transition:
            finished_activities_new = current.finished_activities.copy()
            finished_activities_new[transition.replace("end_", "")] = (
                current.started_activities[transition.replace("end_", "")]
                + self.rcpsp.activities_names_durations[transition.replace("end_", "")]
            )
        return finished_activities_new, started_activities_new

    def all_dependencies_done(self, current, act):
        return (
            len(
                set(self.rcpsp.backward_dependencies.get(act, []))
                - set(current.started_activities.keys())
            )
            == 0
        )

    def solve_branch_and_bound(self, search_strategy="best", width=None, logging=True):
        """
        Solve using branch and bound with different search strategies, all guaranteed to find optimal solution.

        Args:
            search_strategy (str): One of "best", "width", or "depth"
            width (int): Width parameter for width-first search (optional)
            logging (bool): Whether to show progress logging
        """
        init_marking, final_marking = self.get_init_marking_and_final_markings()
        initial_lower_bound = self.heuristic_function(
            [], current_time=0, finished_activities=()
        )

        root = BranchAndBoundNode(
            places_vector=init_marking,
            started_activities={},
            finished_activities={},
            lower_bound=initial_lower_bound,
        )

        if search_strategy == "width":
            # For width-first, we'll use a priority queue with (lower_bound, level, counter)
            priority_queue = [(initial_lower_bound, 0, 0, root)]
        elif search_strategy == "depth":
            # For depth-first, we'll use (lower_bound, level, counter)
            # but process nodes in batches at each level
            priority_queue = [(initial_lower_bound, 0, 0, root)]
        else:  # best-first
            priority_queue = [root]

        best_solution = None
        best_makespan = float("inf")
        nodes_generated = 0
        nodes_expanded = 0
        closed_set = set()
        node_counter = 0

        if logging:
            from tqdm import tqdm

            progress_bar = tqdm(desc=f"Processing ({search_strategy}-first search)")

        while priority_queue:
            if search_strategy == "best":
                current_node = heapq.heappop(priority_queue)
                current_lower_bound = current_node.lower_bound
                current_level = len(current_node.finished_activities)
            else:  # width and depth both use the same tuple structure
                current_lower_bound, current_level, _, current_node = heapq.heappop(
                    priority_queue
                )

            if current_lower_bound >= best_makespan:
                continue

            if current_node in closed_set:
                continue

            closed_set.add(current_node)
            nodes_expanded += 1

            if np.array_equal(current_node.places_vector, final_marking):
                current_makespan = safe_max(current_node.started_activities.values())
                if current_makespan < best_makespan:
                    best_makespan = current_makespan
                    best_solution = current_node
                continue

            neighbors = self.get_neighbors_bb(current_node)
            valid_neighbors = []
            next_level = current_level + 1

            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                nodes_generated += 1
                neighbor_lower_bound = neighbor.lower_bound

                if neighbor_lower_bound < best_makespan:
                    valid_neighbors.append((neighbor_lower_bound, neighbor))

            # Handle different search strategies
            if search_strategy == "width":
                # Sort by lower bound and limit width if specified
                valid_neighbors.sort(key=lambda x: x[0])
                if width is not None:
                    valid_neighbors = valid_neighbors[:width]

                for neighbor_bound, neighbor in valid_neighbors:
                    node_counter += 1
                    heapq.heappush(
                        priority_queue,
                        (neighbor_bound, next_level, node_counter, neighbor),
                    )

            elif search_strategy == "depth":
                # For depth-first:
                # 1. Sort by lower bound first
                # 2. Process all nodes at current level before moving deeper
                # 3. Within each level, explore best lower bounds first
                valid_neighbors.sort(key=lambda x: x[0])  # Sort by lower bound

                for neighbor_bound, neighbor in valid_neighbors:
                    node_counter += 1
                    # Use composite priority that considers both depth and bound
                    # Scale the lower bound to make depth the primary factor
                    # But still maintain proper ordering within each level
                    priority = neighbor_bound + (
                        1000000 * next_level
                    )  # Large multiplier to prioritize depth
                    heapq.heappush(
                        priority_queue, (priority, next_level, node_counter, neighbor)
                    )

            else:  # best-first
                for _, neighbor in valid_neighbors:
                    heapq.heappush(priority_queue, neighbor)

            if logging:
                progress_bar.update(1)
                if nodes_expanded % 1000 == 0:
                    print(
                        f"Nodes expanded: {nodes_expanded}, Best makespan: {best_makespan}"
                    )
                    print(
                        f"Current lower bound: {current_lower_bound}, Level: {current_level}"
                    )
                    print(f"Queue size: {len(priority_queue)}")

        if logging:
            progress_bar.close()

        if best_solution:
            return {
                "scheduling": best_solution.started_activities,
                "total_jobs_scheduled": len(best_solution.started_activities),
                "makespan": best_makespan,
                "nodes_expanded": nodes_expanded,
                "nodes_generated": nodes_generated,
                "search_strategy": search_strategy,
                "solved": True,
            }
        else:
            return {"solved": False}

    def get_available_indices(self, current):
        # optional_starts = [
        #     "start_" + act
        #     for act in self.rcpsp.activities_names_durations
        #     if self.all_dependencies_done(current, act)
        # ]
        # optional_ends = [
        #     "end_" + act
        #     for act in self.rcpsp.activities_names_durations
        #     if act in current.started_activities.keys()
        #     and act not in current.finished_activities.keys()
        # ]
        #
        options = self.matrix_values + current.places_vector[:, np.newaxis]
        non_negative_mask = np.all(options >= 0, axis=0)

        # Find indices of columns where all values are non-negative
        non_negative_indices = np.where(non_negative_mask)[0]

        # Extract corresponding non-negative columns
        non_negative_cols = options[:, non_negative_mask]

        return non_negative_indices, non_negative_cols


import heapq
from tqdm import tqdm


class BranchAndBoundNode:
    def __init__(
        self,
        places_vector,
        started_activities,
        finished_activities,
        lower_bound,
    ):
        self.places_vector = places_vector
        self.started_activities = started_activities
        self.finished_activities = finished_activities
        self.lower_bound = lower_bound

    def __lt__(self, other):
        return (
            self.lower_bound < other.lower_bound
            or (
                self.lower_bound == other.lower_bound
                and len(self.started_activities) > len(other.started_activities)
            )
            or (
                self.lower_bound == other.lower_bound
                and len(self.started_activities) == len(other.started_activities)
                and len(self.finished_activities) > len(other.finished_activities)
            )
        )

    def __eq__(self, other):
        return np.array_equal(self.places_vector, other.places_vector)

    def __hash__(self):
        return hash(self.places_vector.tobytes())


def init_small_problem():
    rcpsp_example = RcpspBase(
        activities_list=[
            {"name": "1", "duration": 3, "resource_demands": {"R1": 2}},
            {"name": "2", "duration": 4, "resource_demands": {"R1": 3}},
            {"name": "3", "duration": 2, "resource_demands": {"R1": 2}},
            {"name": "4", "duration": 6, "resource_demands": {"R1": 4}},
            {"name": "5", "duration": 0, "resource_demands": {}},  # Dummy
        ],
        dependencies={"1": ["3"], "2": ["4"], "3": ["5"], "4": ["5"]},
        resources={"R1": 5},
    )
    matrix = from_rcpsp_to_incidence_matrix(rcpsp_example)
    return matrix, rcpsp_example


def solve_small_problem_math():
    matrix, rcpsp_example = init_small_problem()
    a_star_opt_solver = AstarOptimizedSolver(
        matrix=matrix,
        rcpsp=rcpsp_example,
        heuristic_function=build_cp_heuristic(rcpsp_example),
    )
    result = a_star_opt_solver.solve_a_star()
    return result


def solve_file_problem_math(
    path, beam_search_size=None, timed_transition=False, logging=False
):
    matrix, rcpsp_example = init_real_problem(path)
    # matrix, rcpsp_example = init_small_problem()
    a_star_opt_solver = AstarOptimizedSolver(
        matrix=matrix,
        rcpsp=rcpsp_example,
        heuristic_function=build_cp_heuristic(rcpsp_example),
    )
    result = a_star_opt_solver.solve_branch_and_bound(search_strategy="depth")
    # p, u, e, c = extract_rcpsp_for_solver(path)
    # petri_example.plot("petri_example_hard.png")

    return result
