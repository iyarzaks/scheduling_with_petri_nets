import itertools

from RCPSP_modeling.rcpsp_base import RcpspBase
from RCPSP_modeling.rcpsp_petri_net import RcpspTimedTransitionPetriNet_As
from a_star_solver import cp_heuristic_as
from algorithms_for_solving_rcpsp.a_star import AStarSolver


def parse_rcpsp_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse number of jobs and resources
    num_jobs, num_resources = map(int, lines[0].strip().split())

    # Parse resource capacities
    resource_capacities = list(map(int, lines[1].strip().split()))

    activities = []
    dependencies = {}

    line_index = 3  # Starting index for activities
    for job_id in range(1, num_jobs + 1):
        # Parse activity data
        activity_data = list(map(int, lines[line_index].strip().split()))
        duration = activity_data.pop(0)
        resource_reqs = activity_data[:num_resources]
        num_successors = activity_data[num_resources]
        successors = activity_data[num_resources + 1 :]

        # Create activity dictionary
        activity = {
            "name": job_id,
            "duration": duration,
            "resource_demands": {f"R{i+1}": v for i, v in enumerate(resource_reqs)},
        }
        activities.append(activity)

        # Add dependencies
        dependencies[str(job_id)] = [str(suc) for suc in successors]

        line_index += 1

    return (
        {f"{i + 1}": v for i, v in enumerate(activities)},
        dependencies,
        {f"R{i+1}": v for i, v in enumerate(resource_capacities)},
    )


# Example usage:


def extract_rcpsp_af(file_path):
    jobs, precedence_relations, resource_availabilities = parse_rcpsp_file(file_path)
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


def parse_file_b(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    parameters = list(map(float, lines[0].strip().split()))
    num_subgraphs = int(lines[1].strip())

    subgraphs = []
    index = 2
    for _ in range(num_subgraphs):
        line = list(map(int, lines[index].strip().split()))
        num_branches = line[0]
        branch_ids = line[1:]
        subgraphs.append({"num_branches": num_branches, "branch_ids": branch_ids})
        index += 1

    activities = {}
    for i, line in enumerate(lines[index:]):
        line = list(map(int, line.strip().split()))
        num_branches = line[0]
        branch_ids = line[1:]
        activities[f"{i+1}"] = {"branch_ids": branch_ids}

    return parameters, num_subgraphs, subgraphs, activities


# def extract_sg(file_path):
#     parameters, num_subgraphs, subgraphs, activities = parse_file_b(file_path)
#     return sub_graph_structure


def init_problem_as(a_path, b_path):
    rcpsp_example = extract_rcpsp_af(a_path)
    parameters, num_subgraphs, subgraphs, activities = parse_file_b(b_path)
    petri_example = RcpspTimedTransitionPetriNet_As(
        rcpsp_example, activities, subgraphs
    )

    return petri_example, rcpsp_example


def solve_file_problem_as(a_path, b_path, beam_search_size=None, logging=False):
    petri_example, rcpsp_example = init_problem_as(a_path, b_path)
    job_finish_activity = rcpsp_example.activities[-1].name
    a_star_solver = AStarSolver(
        petri_example,
        rcpsp_example,
        heuristic_function=cp_heuristic_as,
        timed_transition=True,
        job_finish_activity=job_finish_activity,
    )
    petri_example.plot("small_linked_example.png")
    result = a_star_solver.solve(beam_search_size=beam_search_size, logging=logging)
    return result


def find_all_paths_by_branches(deep_precedence_relations, jobs_branch_data, subgraphs):
    all_paths = []
    constant_jobs = [
        job for job in jobs_branch_data if jobs_branch_data[job]["branch_ids"] == [1]
    ]
    combinations = list(itertools.product(*[sg["branch_ids"] for sg in subgraphs]))
    for combination in combinations:
        path_jobs = [
            job
            for job in jobs_branch_data
            if len(set(jobs_branch_data[job]["branch_ids"]).intersection(combination))
            > 0
        ]
        path_jobs = path_jobs + constant_jobs
        path_jobs = sorted(path_jobs, key=lambda x: int(x))
        all_paths.append(path_jobs.copy())
        for pair in itertools.combinations(path_jobs, 2):
            if pair not in deep_precedence_relations:
                second_index = path_jobs.index(pair[1])
                path_jobs[path_jobs.index(pair[0])] = pair[1]
                path_jobs[second_index] = pair[0]
                all_paths.append(path_jobs)

    return all_paths


# def from_all_path_to_log(all_paths):


def from_files_to_log(a_path, b_path):
    _, _, subgraphs, jobs_branch_data = parse_file_b(b_path)
    _, precedence_relations, resource_availabilities = parse_rcpsp_file(a_path)
    deep_precedence_relations = RcpspBase.generate_dependency_matrix(
        precedence_relations
    )
    paths = find_all_paths_by_branches(
        deep_precedence_relations, jobs_branch_data, subgraphs
    )

    print(paths)


def main():
    # df = pd.read_csv(
    #     "/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/results/summary_3.csv"
    # )
    # print("a")
    # from_files_to_log(
    #     a_path="RCPSPAS_data_Servranckx_and_Vanhoucke_EJOR_2019 2/Instances/file0a.RCP",
    #     b_path="RCPSPAS_data_Servranckx_and_Vanhoucke_EJOR_2019 2/Instances/file0b.RCP",
    # )
    result = solve_file_problem_as(
        a_path="RCPSPAS_data_Servranckx_and_Vanhoucke_EJOR_2019 2/dummy_instance/linked_non_nested_example_a.RCP",
        b_path="RCPSPAS_data_Servranckx_and_Vanhoucke_EJOR_2019 2/dummy_instance/linked_non_nested_example_b.RCP",
        logging=True,
    )
    print(result)


if __name__ == "__main__":
    main()
