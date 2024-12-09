import cProfile
import io
import pstats
from collections import deque, defaultdict
from functools import wraps
from itertools import combinations

import numpy as np


def calc_cumulative_resources(resource_over_time):
    # Get all the times in sorted order
    times = sorted(resource_over_time.keys())
    cumulative_resource_usage = {}

    # Initialize cumulative sums to zero for all resources
    total_resources = {}

    # Traverse backwards to accumulate the resource needs
    for time in reversed(times):
        if time not in cumulative_resource_usage:
            cumulative_resource_usage[time] = {}

        for resource, amount in resource_over_time.get(time, {}).items():
            # Add the current resource amount to the cumulative sum
            total_resources[resource] = total_resources.get(resource, 0) + amount

        # Store the cumulative resources at this time point
        cumulative_resource_usage[time] = total_resources.copy()

    return cumulative_resource_usage


def calc_cumulative_with_max(resource_over_time, resource_amounts):
    # Step 1: Calculate cumulative resources over time
    cumulative_resource_usage = calc_cumulative_resources(resource_over_time)

    result_with_max = {}

    # Step 2: For each time, calculate the required sum
    for time, resources in cumulative_resource_usage.items():
        # Step 3: Find max(values.values()) / resource_amount
        max_value = 0
        for resource, amount in resources.items():
            if resource in resource_amounts:
                resource_ratio = amount / resource_amounts[resource]
                max_value = max(max_value, resource_ratio)

        # Step 4: Add the time and max_value to get the final result
        result_with_max[time] = time + np.ceil(max_value)

    return result_with_max


def divide_dicts(d1, d2):
    result = {}
    for key in d1:
        if (
            key in d2 and d2[key] != 0
        ):  # Check if key exists in d2 and avoid division by zero
            result[key] = d1[key] / d2[key]
        else:
            result[key] = (
                None  # Assign None or any other value to indicate division is not possible
            )
    return result


def compute_resource_overages(resource_requirements, total_resources):
    # Initialize a dictionary to hold the overages for each resource
    overages = {resource_name: 0 for resource_name in total_resources}
    time_units_overage = 0

    # Iterate through each time unit and compare resource requirements with available resources
    for t, resources in resource_requirements.items():
        for resource_name, required_amount in resources.items():
            if required_amount > total_resources.get(resource_name, 0):
                overage = required_amount - total_resources.get(resource_name, 0)
                overages[resource_name] += overage
                time_units_overage += 1

    return time_units_overage, overages


def compute_resource_requirements_at_start(schedule, activities):
    # Initialize a dictionary to hold the resource requirements at each start time
    resource_requirements_at_start = {}

    # Iterate through each activity and its start time
    for activity, start_time in schedule.items():
        resources = activities[int(activity) - 1].resource_demands

        # Check if there are any resource demands for this activity
        if resources:
            # If there are no jobs already recorded for this start time, initialize an empty dict
            if start_time not in resource_requirements_at_start:
                resource_requirements_at_start[start_time] = {}

            # Add the resource requirements for this job at its start time
            for resource_name, amount_needed in resources.items():
                if resource_name not in resource_requirements_at_start[start_time]:
                    resource_requirements_at_start[start_time][resource_name] = 0
                resource_requirements_at_start[start_time][resource_name] += (
                    amount_needed * activities[int(activity) - 1].duration
                )

    # Filter out any times where the resource requirements dictionary is empty
    return {time: reqs for time, reqs in resource_requirements_at_start.items() if reqs}


def compute_resource_requirements(schedule, activities):
    # Determine the end time of the project
    end_time = max(
        start_time + activities[int(activity) - 1].duration
        for activity, start_time in schedule.items()
    )

    # Initialize a dictionary to hold the resource requirements for each time unit
    resource_requirements = {t: {} for t in range(end_time + 1)}

    # Iterate through each activity and its start time
    for activity, start_time in schedule.items():
        duration = activities[int(activity) - 1].duration
        resources = activities[int(activity) - 1].resource_demands

        # Add the resource requirement for each time unit the activity is active
        for t in range(start_time, start_time + duration):
            for resource_name, amount_needed in resources.items():
                if resource_name not in resource_requirements[t]:
                    resource_requirements[t][resource_name] = 0
                resource_requirements[t][resource_name] += amount_needed

    return resource_requirements


class Activity:
    def __init__(self, name: str, duration: int, resource_demands: dict):
        self.name = name
        self.duration = duration
        self.resource_demands = {k: v for k, v in resource_demands.items() if v > 0}

        # attributes for CPM calculation:
        self.early_start = 0
        self.early_finish = 0
        self.late_start = 0
        self.late_finish = 0

        # attribute for GRPW calculation
        # self.grpw = 0


def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result

    return wrapper


class RcpspBase:
    def __init__(self, activities_list: list, dependencies: dict, resources: dict):
        self.activities = [
            Activity(a["name"], a["duration"], a["resource_demands"])
            for a in activities_list
        ]
        self.activities_names_durations = {a.name: a.duration for a in self.activities}
        self.activities_names_set = set(self.activities_names_durations.keys())
        self.dependencies = dependencies
        self.resources = resources
        self.backward_dependencies = self.update_backward()
        self.dependencies_deep_set = RcpspBase.generate_dependency_matrix(
            self.dependencies
        )
        self.excuted_memo = {}
        self.memo = {}

    # def depends_on(self, activity, target):
    #     return self.dependencies_deep.get((activity, target), False)

    @staticmethod
    def get_all_subsets(fullset):
        return [
            list(subset)
            for i in range(len(fullset) + 1)
            for subset in combinations(fullset, i)
        ]

    # Example usage
    # def get_all_critical_subset_dict(self):
    #     all_activity_subsets = RcpspBase.get_all_subsets(
    #         [act.name for act in self.activities]
    #     )
    #
    #     # Calculate critical path durations for all subsets
    #     critical_path_durations = {}
    #     for subset in all_activity_subsets:
    #         cp_duration = RcpspBase.find_critical_path_with_executed(
    #             [{act.name: act.duration} for act in self.activities],
    #             self.dependencies,
    #             subset,
    #         )
    #         critical_path_durations[tuple(subset)] = cp_duration
    #         print(f"Executed: {subset} -> Critical Path Duration: {cp_duration}")

    def get_all_critical_path_of_sub(self, executed, job_finish_activity, ongoing):
        return self.find_critical_path_with_executed(
            self.activities_names_durations,
            self.dependencies,
            executed,
            job_finish_activity,
            ongoing,
        )

    @staticmethod
    def remove_special_conditions(dependencies, special_or):
        for key, conditions in special_or.items():
            for condition in conditions:
                if condition in dependencies:
                    if key in dependencies[condition]:
                        dependencies[condition].remove(key)
        return dependencies

    def get_all_critical_path_of_sub_as(
        self, executed, job_finish_activity, or_dependencies
    ):
        if tuple(executed) in self.excuted_memo:
            return self.excuted_memo[tuple(executed)]
        self.dependencies = RcpspBase.remove_special_conditions(
            self.dependencies, or_dependencies
        )
        res = self.find_critical_path_with_executed_as(
            self.activities_names_durations,
            self.dependencies,
            executed,
            job_finish_activity,
            or_dependencies,
        )
        self.excuted_memo[tuple(executed)] = res
        return res

    def calc_cumulative_resources(resource_over_time):
        # Get all the times in sorted order
        times = sorted(resource_over_time.keys())
        cumulative_resource_usage = {}

        # Initialize cumulative sums to zero for all resources
        total_resources = {}

        # Traverse backwards to accumulate the resource needs
        for time in reversed(times):
            if time not in cumulative_resource_usage:
                cumulative_resource_usage[time] = {}

            for resource, amount in resource_over_time.get(time, {}).items():
                # Add the current resource amount to the cumulative sum
                total_resources[resource] = total_resources.get(resource, 0) + amount

            # Store the cumulative resources at this time point
            cumulative_resource_usage[time] = total_resources.copy()

        return cumulative_resource_usage

    def find_critical_path_with_executed(
        self, activities, dependencies, executed, job_finish_activity, ongoing
    ):
        executed_key = (frozenset(executed), frozenset(ongoing.items()))

        if job_finish_activity in executed:
            return (0, [])  # Job is already finished
        if executed_key in self.memo:
            return self.memo[executed_key]

        graph = defaultdict(list)
        in_degree = defaultdict(int)
        duration = activities.copy()

        # Improved handling of ongoing activities
        for job, remaining in ongoing.items():
            duration[job] = remaining
            if job not in activities:
                activities[job] = remaining

        # Build the graph and in-degree dictionary
        for u, neighbors in dependencies.items():
            for v in neighbors:
                graph[u].append(v)
                in_degree[v] += 1
            if u not in in_degree:
                in_degree[u] = 0  # Ensure all nodes are in in_degree

        # Remove executed activities from the graph
        for act in executed:
            if act in graph:
                for neighbor in graph[act]:
                    in_degree[neighbor] -= 1
                del graph[act]
            in_degree.pop(act, None)
            duration.pop(act, None)
            for source in list(graph):
                if act in graph[source]:
                    graph[source].remove(act)

        # Topological sort with cycle detection
        topo_sort = []
        queue = deque([node for node in in_degree if in_degree[node] == 0])
        visited = set()

        while queue:
            if len(topo_sort) > len(activities) - len(executed):
                raise ValueError("Cycle detected in the dependency graph")
            node = queue.popleft()
            topo_sort.append(node)
            visited.add(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_sort) != len(set(activities) - set(executed)):
            raise ValueError("Graph is not a DAG or some nodes are unreachable")

        # Calculate earliest start and latest finish times
        earliest_start = {node: 0 for node in duration}
        latest_finish = {}

        for node in topo_sort:
            for neighbor in graph[node]:
                earliest_start[neighbor] = max(
                    earliest_start[neighbor], earliest_start[node] + duration[node]
                )

        critical_path_duration = max(
            earliest_start[node] + duration[node]
            for node in topo_sort
            if node in duration
        )

        # Calculate latest finish times
        for node in reversed(topo_sort):
            if (
                not graph[node] or node not in duration
            ):  # If it's a sink node or not in duration
                latest_finish[node] = critical_path_duration
            else:
                latest_finish[node] = min(
                    latest_finish[neighbor] - duration[node]
                    for neighbor in graph[node]
                    if neighbor in latest_finish
                )

        # Identify critical path activities
        critical_path_activities = set()
        for node in topo_sort:
            if (
                node in duration
                and earliest_start[node] + duration[node] == latest_finish[node]
            ):
                critical_path_activities.add(node)

        # Refined handling of ongoing activities in critical path
        for job in ongoing:
            if job in critical_path_activities:
                for neighbor in graph[job]:
                    if earliest_start[neighbor] == earliest_start[job] + duration[job]:
                        critical_path_activities.add(neighbor)

        self.memo[executed_key] = (critical_path_duration, critical_path_activities)
        return self.memo[executed_key]

    @staticmethod
    def remove_executed_activities(
        durations, dependencies, special_dependencies, executed_activities
    ):
        # Remove executed activities from durations
        durations = {k: v for k, v in durations.items() if k not in executed_activities}

        # Remove executed activities from dependencies
        dependencies = {
            k: [dep for dep in v if dep not in executed_activities]
            for k, v in dependencies.items()
            if k not in executed_activities
        }

        # Remove executed activities from special dependencies
        special_dependencies = {
            k: v
            for k, v in special_dependencies.items()
            if k not in executed_activities
            and len(set(v).intersection(set(executed_activities))) == 0
        }

        return durations, dependencies, special_dependencies

    @staticmethod
    def calculate_critical_path_duration(
        durations, and_dependencies, or_conditions, job_finish_activity
    ):
        def calculate_earliest_start(activity, memo):
            if activity in memo:
                return memo[activity]
            # if activity == "122":
            #     print("now")

            # Calculate the earliest start based on 'and' dependencies
            dependent_activities = [
                act for act, deps in and_dependencies.items() if activity in deps
            ]
            and_start_times = [
                calculate_earliest_start(dep, memo) + durations[dep]
                for dep in dependent_activities
            ]
            and_start_time = max(and_start_times) if and_start_times else 0

            # Calculate the earliest start based on 'or' conditions
            if activity in or_conditions:
                or_start_times = [
                    calculate_earliest_start(dep, memo) + durations[dep]
                    for dep in or_conditions[activity]
                ]
                or_start_time = min(or_start_times)
            else:
                or_start_time = 0

            # The earliest start time is the maximum of and_start_time and or_start_time
            earliest_start = max(and_start_time, or_start_time)
            memo[activity] = earliest_start
            return earliest_start

        memo = {}
        critical_path_duration = calculate_earliest_start(job_finish_activity, memo)
        return critical_path_duration

    def find_critical_path_with_executed_as(
        self, activities, dependencies, executed, job_finish_activity, or_dependencies
    ):

        executed_key = frozenset(executed)
        if job_finish_activity in executed:
            return 0

        # If the result for the current set of executed tasks is already computed, return it
        if executed_key in self.memo:
            return self.memo[executed_key]

        activities, dependencies, or_dependencies = (
            RcpspBase.remove_executed_activities(
                activities, dependencies, or_dependencies, executed
            )
        )
        if len(activities) < 2:
            return 0

        # Step 1: Create a graph and calculate in-degrees of nodes
        return RcpspBase.calculate_critical_path_duration(
            activities, dependencies, or_dependencies, job_finish_activity
        )

    def update_backward(self):
        backward_dependencies = {}
        for key, values in self.dependencies.items():
            for value in values:
                backward_dependencies.setdefault(value, []).append(key)
        return backward_dependencies

    @staticmethod
    def depends_on_calc(activity, target, dependencies, memo):
        """
        Check if 'activity' depends on 'target' or any of its successors, using memoization.

        :param activity: The activity to check dependencies for.
        :param target: The target activity to check if 'activity' depends on.
        :param dependencies: A dictionary where keys are activities and values are lists of dependent activities.
        :param memo: A dictionary to store the results of previous computations.
        :return: True if 'activity' depends on 'target' or any of its successors, False otherwise.
        """
        if (activity, target) in memo:
            return memo[(activity, target)]

        def dfs(current, target, visited):
            if current == target:
                return True
            if current in visited:
                return False
            visited.add(current)
            for successor in dependencies.get(current, []):
                if dfs(successor, target, visited):
                    return True
            return False

        result = dfs(activity, target, set())
        memo[(activity, target)] = result
        return result

    @staticmethod
    def generate_dependency_matrix(dependencies):
        """
        Generate a dependency matrix indicating if each activity depends on another, using memoization.

        :param dependencies: A dictionary where keys are activities and values are lists of dependent activities.
        :return: A dictionary with tuples of (activity, target) as keys and dependency status (True/False) as values.
        """
        activities = list(dependencies.keys())
        # dependency_matrix = {}
        memo = {}
        dependency_set = set()
        for activity in activities:
            for target in activities:
                if activity != target:
                    if RcpspBase.depends_on_calc(activity, target, dependencies, memo):
                        dependency_set.add((activity, target))

        return dependency_set

    # Example usage:

    # Print the results

    def update_problem(self, removed_activities):

        self.activities = [
            activity
            for activity in self.activities
            if activity.name not in removed_activities
        ]
        self.dependencies = {
            key: value
            for key, value in self.dependencies.items()
            if key not in removed_activities
        }
        self.backward_dependencies = self.update_backward()

    # CPM methods:
    def calculate_early_times(self):
        visited = set()
        for activity in self.activities:
            self._calculate_early_time(activity, visited)

    def _calculate_early_time(
        self, activity, visited
    ):  # calculate activitiy's earlt start+finish
        if activity in visited:
            return
        visited.add(activity)
        max_early_finish = 0
        for predecessor in self.backward_dependencies.get(activity.name, []):
            pred_activity = self.find_activity_by_name(predecessor)
            self._calculate_early_time(pred_activity, visited)
            max_early_finish = max(max_early_finish, pred_activity.early_finish)
        activity.early_start = max_early_finish
        activity.early_finish = max_early_finish + activity.duration

    def calculate_late_times(self):
        try:
            max_late_finish = max(
                (activity.early_finish for activity in self.activities)
            )
        except Exception:
            print("here")
        for activity in self.activities:
            activity.late_finish = max_late_finish

        visited = set()
        for activity in self.activities:
            self._calculate_late_time(activity, visited)

    def _calculate_late_time(self, activity, visited):
        if activity in visited:
            return
        visited.add(activity)
        min_late_start = activity.late_finish - activity.duration
        for successor in self.dependencies.get(activity.name, []):
            succ_activity = self.find_activity_by_name(successor)
            self._calculate_late_time(succ_activity, visited)
            min_late_start = min(
                min_late_start, succ_activity.late_start - activity.duration
            )
        activity.late_start = min_late_start
        activity.late_finish = min_late_start + activity.duration

    def calculate_critical_path(self):
        if len(self.activities) == 0:
            return {
                "duration": 0,
                "critical_path_activities": [],
            }
        cpm_duration = 0
        current_critical = None
        cp = []
        self.calculate_early_times()
        self.calculate_late_times()
        # find the first critical activity:
        for activity in self.activities:
            if activity.early_start == activity.late_start:
                cpm_duration = activity.late_finish
                cp.append(activity.name)
        # for current_critical in cp:
        #     while True:
        #         if current_critical.name not in self.dependencies.keys():
        #             break
        #         else:
        #             for successor in self.dependencies.get(current_critical.name):
        #                 successor_activity = self._find_activity_by_name(successor)
        #                 if (
        #                     successor_activity.early_start
        #                     == successor_activity.late_start
        #                 ):
        #                     current_critical = successor_activity

        return {
            "duration": cpm_duration,
            "critical_path_activities": cp,
        }  # note: returns the cpm duration and A critical path, might be others

    def find_activity_by_name(self, name):
        for activity in self.activities:
            if activity.name == name:
                return activity
        return None

    # GRPW methods:
    # def grpw_calc(self):
