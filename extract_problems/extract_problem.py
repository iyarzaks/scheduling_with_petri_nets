import re

from RCPSP_modeling.rcpsp_base import RcpspBase


def read_file(filename):
    with open(filename, "r") as file:
        data = file.read()
    return data


def extract_table_data(data, start_pattern, end_pattern):
    start_index = data.find(start_pattern) + len(start_pattern)
    end_index = data.find(end_pattern, start_index)
    table_data = data[start_index:end_index].strip()
    return table_data.split("\n")


def extract_opt_values(text_file):
    with open(text_file, "r") as file:
        data = file.readlines()

    # Initialize a dictionary to store makespan values
    makespan_values = {}

    # Iterate through the lines in the file
    for line in data:
        # Check if the line contains parameter, instance, and makespan
        if line.strip().replace(" ", "").replace(".", "").isdigit():
            param, instance, makespan, time = line.split()
            makespan_values[(param, instance)] = int(makespan)
    return makespan_values


def extract_opt_values_with_time(text_file):
    with open(text_file, "r") as file:
        data = file.readlines()

    # Initialize a dictionary to store makespan values
    makespan_values = []

    # Iterate through the lines in the file
    for line in data:
        # Check if the line contains parameter, instance, and makespan
        if line.strip().replace(" ", "").replace(".", "").isdigit():
            param, instance, makespan, time = line.split()
            makespan_values.append(
                {
                    "param": param,
                    "instance": instance,
                    "makespan": makespan,
                    "time": time,
                }
            )
    return makespan_values


def parse_precedence_relations(table_data):
    dependencies = {}
    for line in table_data[1:]:
        if line.strip() != "":
            parts = line.split()
            if len(parts) > 1:
                jobnr = parts[0]
                successors = [x for x in parts[3:]]
                dependencies[jobnr] = successors
    return dependencies


def parse_resource_availabilities(table_data):
    resources = {}
    headers = table_data[0].split("  ")
    for line in table_data[1:]:
        parts = line.split()
        if len(parts) > 1:
            for i in range(len(headers)):
                if headers[i].startswith("R"):
                    resource_name = headers[i]
                    resource_amount = int(parts[i])
                    resources[resource_name.replace(" ", "")] = resource_amount
    return resources


def parse_requests_durations(table_data):
    jobs = {}
    for line in table_data[1:]:
        if line.strip() != "":
            parts = line.split()
            if len(parts) > 1:
                jobnr = parts[0]
                duration = int(parts[2])
                resource_demands = {
                    f"R{i+1}": int(parts[i + 3]) for i in range(0, len(parts) - 3)
                }
                jobs[jobnr] = {
                    "duration": duration,
                    "resource_demands": resource_demands,
                }
    return jobs


def extract_rcpsp(filename):
    data = read_file(filename)

    precedence_relations_pattern = "PRECEDENCE RELATIONS:"
    precedence_relations_end_pattern = "REQUESTS/DURATIONS:"
    precedence_relations_data = extract_table_data(
        data, precedence_relations_pattern, precedence_relations_end_pattern
    )

    precedence_relations = parse_precedence_relations(precedence_relations_data)

    requests_durations_pattern = "REQUESTS/DURATIONS:"
    requests_durations_end_pattern = "RESOURCEAVAILABILITIES:"
    requests_durations_data = extract_table_data(
        data, requests_durations_pattern, requests_durations_end_pattern
    )

    jobs = parse_requests_durations(requests_durations_data)

    resource_availabilities_pattern = "RESOURCEAVAILABILITIES:"
    resource_availabilities_end_pattern = (
        "************************************************************************"
    )
    resource_availabilities_data = extract_table_data(
        data, resource_availabilities_pattern, resource_availabilities_end_pattern
    )
    resource_availabilities = parse_resource_availabilities(
        resource_availabilities_data
    )
    return RcpspBase(
        activities_list=[
            {
                "name": str(k),
                "duration": v["duration"],
                "resource_demands": v["resource_demands"],
            }
            for k, v in jobs.items()
        ],
        dependencies=precedence_relations,
        resources=resource_availabilities,
    )


def extract_rcpsp_for_solver(filename):
    data = read_file(filename)

    durations = []
    resource_consumption = []
    precedence_constraints = {}
    available_resources = []

    # Extract durations and resource consumption
    requests_section = re.search(
        r"REQUESTS/DURATIONS:(.*?)RESOURCEAVAILABILITIES:", data, re.DOTALL
    ).group(1)
    for line in requests_section.splitlines():
        match = re.match(
            r"\s*(\d+)\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line
        )
        if match:
            jobnr, duration, r1, r2, r3, r4 = map(int, match.groups())
            durations.append(duration)
            resource_consumption.append((jobnr, [r1, r2, r3, r4]))

    # Extract precedence constraints
    precedence_section = re.search(
        r"PRECEDENCE RELATIONS:(.*?)REQUESTS/DURATIONS:", data, re.DOTALL
    ).group(1)
    for line in precedence_section.splitlines():
        match = re.match(r"\s*(\d+)\s+\d+\s+\d+\s+(.*)", line)
        if match:
            jobnr = int(match.group(1))
            successors = list(map(int, match.group(2).split()))
            precedence_constraints[jobnr] = successors

    # Extract available resources
    resources_section = re.search(
        r"RESOURCEAVAILABILITIES:(.*?)\*", data, re.DOTALL
    ).group(1)
    for line in resources_section.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
        if match:
            available_resources = list(map(int, match.groups()))

    resource_consumption = [r[1] for r in resource_consumption]
    precedence_constraints_new = []
    for key in precedence_constraints:
        for val in precedence_constraints[key]:
            precedence_constraints_new.append([key - 1, val - 1])

    return (
        durations,
        resource_consumption,
        precedence_constraints_new,
        available_resources,
    )
