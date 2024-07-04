import cProfile
import io
import pstats
from collections import deque, defaultdict
from functools import wraps
from itertools import combinations


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
        self.dependencies = dependencies
        self.resources = resources
        self.backward_dependencies = self.update_backward()
        self.dependencies_deep = RcpspBase.generate_dependency_matrix(self.dependencies)
        self.dependencies_deep_set = RcpspBase.generate_dependency_matrix(
            self.dependencies
        )

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
    def get_all_critical_subset_dict(self):
        all_activity_subsets = RcpspBase.get_all_subsets(
            [act.name for act in self.activities]
        )

        # Calculate critical path durations for all subsets
        critical_path_durations = {}
        for subset in all_activity_subsets:
            cp_duration = RcpspBase.find_critical_path_with_executed(
                [{act.name: act.duration} for act in self.activities],
                self.dependencies,
                subset,
            )
            critical_path_durations[tuple(subset)] = cp_duration
            print(f"Executed: {subset} -> Critical Path Duration: {cp_duration}")

    def get_all_critical_path_of_sub(self, executed):
        return self.find_critical_path_with_executed(
            self.activities_names_durations,
            self.dependencies,
            executed,
        )

    def find_critical_path_with_executed(self, activities, dependencies, executed):
        executed_key = frozenset(executed)

        # If the result for the current set of executed tasks is already computed, return it
        if executed_key in self.memo:
            return self.memo[executed_key]

        # Step 1: Create a graph and calculate in-degrees of nodes
        graph = defaultdict(list)
        in_degree = {activity: 0 for activity in activities}
        duration = {activity: activities[activity] for activity in activities}

        for u in dependencies:
            for v in dependencies[u]:
                graph[u].append(v)
                in_degree[v] += 1

        # Remove executed activities and update graph
        for act in executed:
            if act in graph:
                for neighbor in graph[act]:
                    in_degree[neighbor] -= 1
                del graph[act]
            if act in in_degree:
                del in_degree[act]
            if act in duration:
                del duration[act]

        # Step 2: Topological sorting using Kahn's Algorithm
        topo_sort = []
        queue = deque([node for node in in_degree if in_degree[node] == 0])

        while queue:
            node = queue.popleft()
            topo_sort.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Step 3: Initialize earliest start times
        earliest_start = {node: 0 for node in duration}

        # Step 4: Compute the earliest start and finish times
        for node in topo_sort:
            for neighbor in graph[node]:
                earliest_start[neighbor] = max(
                    earliest_start[neighbor], earliest_start[node] + duration[node]
                )

        # Step 5: Find the maximum finish time which is the critical path duration
        critical_path_duration = (
            max(earliest_start[node] + duration[node] for node in duration)
            if duration
            else 0
        )

        # Cache the result
        self.memo[executed_key] = critical_path_duration

        return critical_path_duration

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
