class Activity:
    def __init__(self, name: str, duration: int, resource_demands: dict):
        self.name = name
        self.duration = duration
        self.resource_demands = resource_demands

        # attributes for CPM calculation:
        self.early_start = 0
        self.early_finish = 0
        self.late_start = 0
        self.late_finish = 0

        # attribute for GRPW calculation
        # self.grpw = 0


class RcpspBase:
    def __init__(self, activities_list: list, dependencies: dict, resources: dict):
        self.activities = [
            Activity(a["name"], a["duration"], a["resource_demands"])
            for a in activities_list
        ]
        self.dependencies = dependencies
        self.resources = resources
        self.backward_dependencies = {}

        for key, values in self.dependencies.items():
            for value in values:
                self.backward_dependencies.setdefault(value, []).append(key)

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
            pred_activity = self._find_activity_by_name(predecessor)
            self._calculate_early_time(pred_activity, visited)
            max_early_finish = max(max_early_finish, pred_activity.early_finish)
        activity.early_start = max_early_finish
        activity.early_finish = max_early_finish + activity.duration

    def calculate_late_times(self):
        max_early_finish = max((activity.early_finish for activity in self.activities))
        for activity in self.activities:
            activity.late_finish = max_early_finish

        visited = set()
        for activity in self.activities:
            self._calculate_late_time(activity, visited)

    def _calculate_late_time(self, activity, visited):
        if activity in visited:
            return
        visited.add(activity)
        min_late_start = float("inf")
        for successor in self.dependencies.get(activity.name, []):
            succ_activity = self._find_activity_by_name(successor)
            self._calculate_late_time(succ_activity, visited)
            min_late_start = min(min_late_start, succ_activity.late_start)
        activity.late_finish = min_late_start
        activity.late_start = min_late_start - activity.duration

    def calculate_critical_path(self):
        cpm_duration = 0
        current_critical = None
        cp = []
        self.calculate_early_times()
        self.calculate_late_times()
        # find the first critical activity:
        for activity in self.activities:
            if activity.early_start == activity.late_start == 0:
                current_critical = activity
        while True:
            cpm_duration += current_critical.duration
            cp.append(current_critical.name)
            if current_critical.name not in self.dependencies.keys():
                break
            else:
                for successor in self.dependencies.get(current_critical.name):
                    if successor.early_start == successor.late_start:
                        current_critical = successor

        return (
            cpm_duration,
            cp,
        )  # note: returns the cpm duration and A critical path, might be others

    def _find_activity_by_name(self, name):
        for activity in self.activities:
            if activity.name == name:
                return activity
        return None

    # GRPW methods:
    # def grpw_calc(self):
