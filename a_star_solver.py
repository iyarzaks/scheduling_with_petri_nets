import json
import os
import re
import signal
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from RCPSP_modeling.rcpsp_base import RcpspBase
from RCPSP_modeling.rcpsp_petri_net import (
    RcpspTimedPlacePetriNet,
    RcpspTimedTransitionPetriNet,
)
from algorithms_for_solving_rcpsp.a_star import AStarSolver
from algorithms_for_solving_rcpsp.solve_with_mip_solver import solve_rcpsp_optimizer
from extract_problems.extract_problem import (
    extract_rcpsp,
    extract_opt_values,
    extract_opt_values_with_time,
)


def cp_heuristic_for_timed_place(
    rcpsp_example: RcpspBase, removed_activities, started_activities, current_time
):
    if started_activities:
        expected_finsh_time = max(
            [
                rcpsp_example.find_activity_by_name(act).duration
                + started_activities[act]
                for act in started_activities
            ]
        )
    else:
        expected_finsh_time = 0
    remaining_time_of_started_jobs = expected_finsh_time - current_time

    rcpsp_example.update_problem(removed_activities=started_activities)
    return max(
        remaining_time_of_started_jobs,
        rcpsp_example.calculate_critical_path()["duration"],
    )


def cp_heuristic(
    rcpsp_example: RcpspBase, removed_activities, started_activities, current_time
):

    if started_activities:
        last_activity = max(removed_activities, key=removed_activities.get)
        rel_activities = [
            act
            for act in rcpsp_example.activities
            if act.name not in started_activities
        ]
        independent_activities = [
            act.name
            for act in rel_activities
            if (last_activity, act.name) not in rcpsp_example.dependencies_deep_set
        ]
        unk_time = current_time - max(started_activities.values())
    else:
        independent_activities = []
        unk_time = 0
    # rcpsp_example.get_all_critical_path_of_sub()
    # opt_1 = copy.copy(rcpsp_example)
    # opt_1.update_problem(
    #     removed_activities=list(started_activities) + independent_activities
    # )
    # opt_2 = copy.copy(rcpsp_example)
    # opt_2.update_problem(removed_activities=list(started_activities))
    return max(
        rcpsp_example.get_all_critical_path_of_sub(
            executed=list(started_activities) + independent_activities
        ),
        rcpsp_example.get_all_critical_path_of_sub(list(started_activities)) - unk_time,
    )


def solver():
    petri_example, rcpsp_example = init_small_example()
    a_star_solver = AStarSolver(
        petri_example, rcpsp_example, heuristic_function=cp_heuristic
    )
    a_star_solver.solve(beam_search_size=25)


def init_real_problem(file_path, timed_transition):
    rcpsp_example = extract_rcpsp(file_path)
    if timed_transition:
        petri_example = RcpspTimedTransitionPetriNet(rcpsp_example)
    else:
        petri_example = RcpspTimedPlacePetriNet(rcpsp_example)
    return petri_example, rcpsp_example


def analyze_results(dir_path):
    true_count = 0
    false_count = 0
    total_opt = 0
    files_not_solved = []
    nodes_visited = []
    total_makespan = 0
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, "r") as file:
            data = json.load(file)
            if "solved" in data:
                if data["solved"]:
                    true_count += 1
                    nodes_visited.append(data["nodes_visited"])
                    total_opt += data["opt_value"]
                    total_makespan += max(data["makespan"], data["opt_value"])
                    # if data["makespan"] > data["opt_value"]:
                    #     print(filepath)
                else:
                    files_not_solved.append(filename)
                    false_count += 1

    print(f"Files with 'solved' = True: {true_count}")
    print(f"Files with 'solved' = False: {false_count}")
    print(f"Mean nodes_visited: {np.mean(nodes_visited)}")
    print(f"Median nodes_visited: {np.median(nodes_visited)}")
    print(f"Mean ratio: {total_makespan / total_opt}")
    return files_not_solved


def summarize_results_to_csv(TT_files, TP_file):
    opt_values = extract_opt_values_with_time(
        "/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j30opt.txt"
    )

    for opt_value in opt_values:

        filepath = os.path.join(
            TT_files, opt_value["param"] + "_" + opt_value["instance"]
        )
        try:
            with open(filepath, "r") as file:
                data = json.load(file)
                if "solved" in data:
                    if data["solved"]:
                        opt_value["timed_transition_solved"] = data["solved"]
                        opt_value["timed_transition_makespan"] = max(
                            data["makespan"], data["opt_value"]
                        )
                        opt_value["timed_transition_nodes_visited"] = data[
                            "nodes_visited"
                        ]
                    else:
                        opt_value["timed_transition_solved"] = data["solved"]
        except:
            pass

        filepath_tp = os.path.join(
            TP_file, opt_value["param"] + "_" + opt_value["instance"]
        )
        with open(filepath_tp, "r") as file:
            data_tp = json.load(file)
            if "solved" in data_tp:
                if data_tp["solved"]:
                    opt_value["timed_place_solved"] = data_tp["solved"]
                    opt_value["timed_place_makespan"] = max(
                        data_tp["makespan"], data_tp["opt_value"]
                    )
                    opt_value["timed_place_nodes_visited"] = data_tp["nodes_visited"]
                else:
                    opt_value["timed_place_solved"] = data_tp["solved"]

    return pd.DataFrame(opt_values)


def init_small_example(timed_transition=False):
    rcpsp_example = RcpspBase(
        activities_list=[
            {"name": "A", "duration": 8, "resource_demands": {"R1": 4}},
            {"name": "B1", "duration": 4, "resource_demands": {"R1": 4}},
            {"name": "B2", "duration": 4, "resource_demands": {"R1": 4}},
            {"name": "B3", "duration": 4, "resource_demands": {"R1": 4}},
            {"name": "B4", "duration": 4, "resource_demands": {"R1": 4}},
            {"name": "B5", "duration": 4, "resource_demands": {"R1": 4}},
            {"name": "B6", "duration": 4, "resource_demands": {"R1": 4}},
            {"name": "C", "duration": 4, "resource_demands": {"R1": 3}},
        ],
        dependencies={
            "A": ["B1", "B2", "B3", "B4", "B5", "B6"],
            "B1": ["C"],
            "B2": ["C"],
            "B3": ["C"],
            "B4": ["C"],
            "B5": ["C"],
            "B6": ["C"],
        },
        resources={"R1": 1000},
    )
    if timed_transition:
        petri_example = RcpspTimedTransitionPetriNet(rcpsp_example)
    else:
        petri_example = RcpspTimedPlacePetriNet(rcpsp_example)
    return petri_example, rcpsp_example


def init_slide_example():
    rcpsp_example = RcpspBase(
        activities_list=[
            {"name": "a", "duration": 0, "resource_demands": {}},
            {"name": "b", "duration": 2, "resource_demands": {"R1": 2}},
            {"name": "c", "duration": 2, "resource_demands": {"R1": 3}},
            {"name": "e", "duration": 0, "resource_demands": {}},
        ],
        dependencies={"a": ["b", "c"], "b": ["e"], "c": ["e"]},
        resources={"R1": 5},
    )
    petri_example = RcpspTimedPlacePetriNet(rcpsp_example)
    return petri_example, rcpsp_example


def check_opt_vs_initial_heuristic():
    opt_values = extract_opt_values(
        "/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j30opt.txt"
    )
    comparison = []
    for file in os.listdir("extract_problems/data/j30.sm.tgz"):
        param = re.search(r"j30(\d+)_", file).group(1)
        instance = file.split("_")[-1].split(".")[0]
        rcpsp_example = extract_rcpsp(
            f"/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j30.sm.tgz/{file}"
        )
        cpm_heuristic = rcpsp_example.calculate_critical_path()["duration"]
        comparison.append(
            {
                "param": param,
                "instance": instance,
                "opt_value": opt_values[(param, instance)],
                "heuristic": cpm_heuristic,
            }
        )
    comparison_df = pd.DataFrame(comparison)
    comparison_df["ratio"] = comparison_df["heuristic"] / comparison_df["opt_value"]
    print(comparison_df["ratio"].mean())
    comparison_df.to_csv("comparison_results_30.csv", index=False)


def solve_file_problem(
    path, beam_search_size=None, timed_transition=False, logging=False
):
    petri_example, rcpsp_example = init_real_problem(
        path, timed_transition=timed_transition
    )
    if timed_transition:
        a_star_solver = AStarSolver(
            petri_example,
            rcpsp_example,
            heuristic_function=cp_heuristic,
            timed_transition=timed_transition,
        )
    else:
        a_star_solver = AStarSolver(
            petri_example,
            rcpsp_example,
            heuristic_function=cp_heuristic_for_timed_place,
            timed_transition=timed_transition,
        )
    result = a_star_solver.solve(beam_search_size=beam_search_size, logging=logging)
    return result


def solve_small_problem(timed_transition=False, beam_search_size=None):
    petri_example, rcpsp_example = init_small_example(timed_transition=timed_transition)
    a_star_solver = AStarSolver(
        petri_example,
        rcpsp_example,
        heuristic_function=cp_heuristic,
        timed_transition=timed_transition,
    )
    result = a_star_solver.solve(beam_search_size=beam_search_size)
    return result


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def run_with_timeout(timeout, func, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
    except TimeoutException:
        result = {"solved": False, "run_time": elapsed_time}
    finally:
        signal.alarm(0)
    result["run_time"] = elapsed_time
    return result


def solve_problem_with_time_limit(
    timeout, problem_file, opt_values, beam_search_size, timed_transition
):
    param = re.search(r"j30(\d+)_", problem_file).group(1)
    instance = problem_file.split("_")[-1].split(".")[0]
    result = run_with_timeout(
        timeout=timeout,
        func=solve_rcpsp_optimizer,
        path=problem_file,
        # beam_search_size=beam_search_size,
        # timed_transition=timed_transition,
    )
    result["opt_value"] = opt_values[(param, instance)]
    with open(
        f"results/j30_time_cbc_solver/{param}_{instance}",
        "w",
    ) as file:
        json.dump(result, file)


def run_over_files(files_not_solved=None):
    opt_values = extract_opt_values("extract_problems/data/j30opt.txt")
    files_to_check = []
    for file in os.listdir("extract_problems/data/j30.sm.tgz"):
        if (
            not os.path.exists(
                f'results/j30_time_cbc_solver/{file.replace("j30", "").replace(".sm", "")}'
            )
            and file.replace("j30", "").replace(".sm", "") in files_not_solved
        ):
            files_to_check.append(file)
    print(f"{len(files_to_check)} files to check")
    progress_bar = tqdm(total=len(files_to_check), desc="Processing")
    for file in files_to_check:

        solve_problem_with_time_limit(
            timeout=600,
            problem_file=f"extract_problems/data/j30.sm.tgz/{file}",
            opt_values=opt_values,
            beam_search_size=None,
            timed_transition=True,
        )
        progress_bar.update(1)


def main():
    # res_df = summarize_results_to_csv(
    #     "results/j30_time_transition_depends_heuristic", "results/j30_no_beam"
    # )
    # res_df.to_csv("results/summary_2.csv", index=False)

    # analyze_results("results/j30_time_transition_depends_heuristic")
    # analyze_results("results/j30")
    easy_problems = pd.read_csv("results/summary_2.csv")
    easy_problems = easy_problems[easy_problems["timed_transition_solved"]]
    easy_problems["param"] = easy_problems["param"].apply(str)
    easy_problems["param"] = easy_problems["param"] + "_"
    easy_problems["instance"] = easy_problems["instance"].apply(str)
    easy_problems["problem_instance"] = (
        easy_problems["param"] + easy_problems["instance"]
    )
    easy_problems = easy_problems["problem_instance"]
    run_over_files(list(easy_problems))
    # print(solve_small_problem(timed_transition=True))
    # print(
    #     solve_file_problem(
    #         path="/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j30.sm.tgz/j3048_5.sm",
    #         timed_transition=True,
    #         beam_search_size=None,
    #         logging=True,
    #     )
    # )


if __name__ == "__main__":
    main()
