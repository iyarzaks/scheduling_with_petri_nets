class Activity:
    def __init__(self, name: str, duration: int, resource_demands: dict):
        self.name = name
        self.duration = duration
        self.resource_demands = resource_demands


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
