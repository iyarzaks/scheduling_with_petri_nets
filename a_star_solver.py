import numpy as np
import pandas as pd

from RCPSP_modeling.rcpsp_base import RcpspBase, divide_dicts
from RCPSP_modeling.rcpsp_petri_net import (
    RcpspTimedPlacePetriNet,
    RcpspTimedTransitionPetriNet,
)
from algorithms_for_solving_rcpsp.a_star import AStarSolver
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


def sum_resource_demands(activities):
    total_demands = {}

    for activity in activities:
        for resource, demand in activity.resource_demands.items():
            if resource in total_demands:
                total_demands[resource] += demand * activity.duration
            else:
                total_demands[resource] = demand

    return total_demands


def resources_heuristic(
    rcpsp_example: RcpspBase,
    removed_activities,
    started_activities,
    current_time,
    job_finish_activity,
    alternatives,
):
    activities_left = [
        act for act in rcpsp_example.activities if act.name not in removed_activities
    ]
    if len(activities_left) < 2:
        return 0
    sum_of_resource_demands = sum_resource_demands(activities_left)
    divides_by_amount = divide_dicts(sum_of_resource_demands, rcpsp_example.resources)
    if not divides_by_amount:
        print("no")
    max_of_divdes = max(divides_by_amount.values())
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
            executed=list(started_activities) + independent_activities,
            job_finish_activity=job_finish_activity,
        ),
        rcpsp_example.get_all_critical_path_of_sub(
            list(started_activities),
            job_finish_activity,
        )
        - unk_time,
        max_of_divdes,
    )


def cp_heuristic(
    rcpsp_example: RcpspBase,
    removed_activities,
    started_activities,
    current_time,
    job_finish_activity,
    alternatives,
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
            executed=list(started_activities) + independent_activities,
            job_finish_activity=job_finish_activity,
            ongoing={},
        )[0],
        rcpsp_example.get_all_critical_path_of_sub(
            list(started_activities), job_finish_activity, ongoing={}
        )[0]
        - unk_time,
    )


def cp_heuristic_as(
    rcpsp_example: RcpspBase,
    removed_activities,
    started_activities,
    current_time,
    job_finish_activity,
    alternatives,
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
    res = 9999999999
    for alternative in alternatives.values():
        # alternative_set = set(alternative)
        # not_relevant = list(set(rcpsp_example.activities_names_durations) - set(alternative))
        # not_relevant = [
        #     act
        #     for act in rcpsp_example.activities_names_durations
        #     if act not in alternative_set
        # ]
        first_res = rcpsp_example.get_all_critical_path_of_sub(
            executed=set(
                list(started_activities) + independent_activities + alternative
            ),
            job_finish_activity=job_finish_activity,
        )
        if first_res < res:
            # print("not calc 2 ")
            # continue

            res_of_alternative = max(
                first_res,
                rcpsp_example.get_all_critical_path_of_sub(
                    set(list(started_activities) + alternative),
                    job_finish_activity,
                )
                - unk_time,
            )
            if res_of_alternative < res:
                res = res_of_alternative

    return res
    # return max(
    #     rcpsp_example.get_all_critical_path_of_sub_as(
    #         executed=list(started_activities) + independent_activities,
    #         job_finish_activity=job_finish_activity,
    #         or_dependencies=or_dependencies.copy(),
    #     ),
    #     rcpsp_example.get_all_critical_path_of_sub_as(
    #         list(started_activities),
    #         job_finish_activity,
    #         or_dependencies=or_dependencies,
    #     )
    #     - unk_time,
    # )


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


def summarize_results_to_csv(results_dir):
    opt_values = extract_opt_values_with_time(
        "/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j30opt.txt"
    )
    for dir in results_dir:
        for opt_value in opt_values:

            filepath = os.path.join(
                dir, opt_value["param"] + "_" + opt_value["instance"]
            )
            try:
                with open(filepath, "r") as file:
                    data = json.load(file)
                    if "solved" in data:
                        if data["solved"]:
                            opt_value[f"{dir}_solved"] = data["solved"]
                            opt_value[f"{dir}_makespan"] = data["makespan"]
                            opt_value[f"{dir}_run_time"] = data["run_time"]
                            if "nodes_expanded" in data:
                                opt_value[f"{dir}_nodes_expanded"] = data[
                                    "nodes_expanded"
                                ]
                            else:
                                opt_value[f"{dir}_nodes_expanded"] = data[
                                    "nodes_expand"
                                ]

                            opt_value[f"{dir}_nodes_generated"] = data[
                                "nodes_generated"
                            ]
                            # opt_value["timed_transition_nodes_visited"] = data[
                            #     "nodes_visited"
                            # ]
                        else:
                            opt_value[f"{dir}_solved"] = False
            except:
                pass

            # filepath_tp = os.path.join(
            #     TP_file, opt_value["param"] + "_" + opt_value["instance"]
            # )
            # with open(filepath_tp, "r") as file:
            #     data_tp = json.load(file)
            #     if "solved" in data_tp:
            #         if data_tp["solved"]:
            #             opt_value["timed_place_solved"] = data_tp["solved"]
            #             opt_value["timed_place_makespan"] = max(
            #                 data_tp["makespan"], data_tp["opt_value"]
            #             )
            #             opt_value["timed_place_nodes_visited"] = data_tp["nodes_visited"]
            #         else:
            #             opt_value["timed_place_solved"] = data_tp["solved"]

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
    path, beam_search_size=None, timed_transition=True, logging=False
):
    petri_example, rcpsp_example = init_real_problem(
        path, timed_transition=timed_transition
    )
    # p, u, e, c = extract_rcpsp_for_solver(path)
    # petri_example.plot("petri_example_hard.png")
    if timed_transition:
        a_star_solver = AStarSolver(
            petri_example,
            rcpsp_example,
            heuristic_function=cp_heuristic,
            timed_transition=timed_transition,
            job_finish_activity=rcpsp_example.activities[-1].name,
            # heuristic_params=(p, u, e, c),
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


def run_with_timeout(timeout, func, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        result["run_time"] = elapsed_time
    except TimeoutException:
        result = {"solved": False, "run_time": timeout}
    finally:
        signal.alarm(0)
    return result


import os
import re
import json
import signal
import time
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
from functools import partial


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def run_with_timeout(timeout, func, *args, **kwargs):
    """
    Cross-platform timeout handler that works on both Unix and Windows.
    Falls back to threading-based timeout on Windows.
    """
    if platform.system() != "Windows":
        # Unix-based systems: use SIGALRM
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            result["run_time"] = elapsed_time
        except TimeoutException:
            result = {"solved": False, "run_time": timeout}
        finally:
            signal.alarm(0)
    else:
        # Windows: use ProcessPoolExecutor with timeout
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            start_time = time.time()
            try:
                result = future.result(timeout=timeout)
                elapsed_time = time.time() - start_time
                result["run_time"] = elapsed_time
            except TimeoutError:
                result = {"solved": False, "run_time": timeout}

    return result


def run_over_files(results_dir, timeout, max_retries=3, retry_delay=5):
    """
    Process files with automatic retries and robust error handling.
    """
    os.makedirs(results_dir, exist_ok=True)
    opt_values = extract_opt_values("extract_problems/data/j30opt.txt")

    # Track failed files for retry
    failed_files = []

    files_to_check = [
        file
        for file in os.listdir("extract_problems/data/j30.sm.tgz")
        if not os.path.exists(
            f'{results_dir}/{file.replace("j30", "").replace(".sm", "")}'
        )
    ]

    print(f"{len(files_to_check)} files to check")

    def process_batch(file_list, attempt=1):
        progress_bar = tqdm(
            total=len(file_list), desc=f"Processing (Attempt {attempt})"
        )

        current_failed = []

        # Use fewer workers to avoid memory issues
        max_workers = min(os.cpu_count() or 1, 6)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            solve_func = partial(
                solve_wrapper,
                results_dir=results_dir,
                timeout=timeout,
                opt_values=opt_values,
            )

            # Submit all tasks at once - the ProcessPoolExecutor will handle the queuing
            futures = {executor.submit(solve_func, file): file for file in file_list}

            # Process results as they complete (faster jobs will finish first)
            for future in as_completed(futures):
                original_file = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        current_failed.append(original_file)
                        print(f"Failed {original_file}: {result}")
                    else:
                        print(f"Completed {original_file}")
                except Exception as e:
                    current_failed.append(original_file)
                    print(f"Error processing {original_file}: {e}")
                progress_bar.update(1)

        progress_bar.close()
        return current_failed

    # Initial processing
    current_files = files_to_check
    attempt = 1

    while attempt <= max_retries and current_files:
        if attempt > 1:
            print(f"\nRetry attempt {attempt} for {len(current_files)} files")
            time.sleep(retry_delay)

        failed_files = process_batch(current_files, attempt)

        # Update for next iteration
        current_files = failed_files
        attempt += 1

    # Final report
    if failed_files:
        print(
            f"\nFailed to process {len(failed_files)} files after {max_retries} attempts:"
        )
        for file in failed_files:
            print(f"- {file}")
    else:
        print("\nAll files processed successfully!")


def solve_wrapper(file, results_dir, timeout, opt_values):
    """Wrapper to catch and handle exceptions, including timeouts."""
    max_attempts = 2  # Individual file retry attempts
    attempt = 0

    while attempt < max_attempts:
        try:
            solve_problem_with_time_limit(
                results_dir=results_dir,
                timeout=timeout,
                problem_file=f"extract_problems/data/j30.sm.tgz/{file}",
                opt_values=opt_values,
                beam_search_size=None,
                timed_transition=True,
            )
            return True
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                return e
            time.sleep(1)  # Short delay between attempts


def solve_problem_with_time_limit(
    results_dir, timeout, problem_file, opt_values, beam_search_size, timed_transition
):
    param = re.search(r"j30(\d+)_", problem_file).group(1)
    instance = problem_file.split("_")[-1].split(".")[0]

    try:
        result = run_with_timeout(
            timeout=timeout,
            func=solve_file_problem,
            path=problem_file,
        )
        result["opt_value"] = opt_values.get((param, instance), None)

        # Ensure atomic write using temporary file
        temp_file = f"{results_dir}/{param}_{instance}.tmp"
        final_file = f"{results_dir}/{param}_{instance}"

        try:
            with open(temp_file, "w") as file:
                json.dump(result, file)
            os.rename(temp_file, final_file)  # Atomic operation
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

    except Exception as e:
        print(f"Error processing {problem_file}: {str(e)}")
        raise e


# def run_over_files(results_dir, timeout):
#     opt_values = extract_opt_values("extract_problems/data/j30opt.txt")
#     files_to_check = []
#     for file in os.listdir("extract_problems/data/j30.sm.tgz"):
#         if not os.path.exists(
#             f'{results_dir}/{file.replace("j30", "").replace(".sm", "")}'
#         ):
#             files_to_check.append(file)
#     print(f"{len(files_to_check)} files to check")
#     progress_bar = tqdm(total=len(files_to_check), desc="Processing")
#     for file in files_to_check:
#
#         solve_problem_with_time_limit(
#             results_dir=results_dir,
#             timeout=timeout,
#             problem_file=f"extract_problems/data/j30.sm.tgz/{file}",
#             opt_values=opt_values,
#             beam_search_size=None,
#             timed_transition=True,
#         )
#         progress_bar.update(1)


def main():
    # res_df = summarize_results_to_csv(
    #     [
    #         "results/paper_version_scip",
    #         "results/paper_version_TTPN_10_min",
    #     ]
    # )
    # res_df.to_csv("results/paper_new_scip.csv", index=False)

    # analyze_results("results/j30_time_transition_depends_heuristic")
    # analyze_results("results/j30")
    # easy_problems = pd.read_csv("results/summary_2.csv")
    # easy_problems = easy_problems[easy_problems["timed_transition_solved"]]
    # easy_problems["param"] = easy_problems["param"].apply(str)
    # easy_problems["param"] = easy_problems["param"] + "_"
    # easy_problems["instance"] = easy_problems["instance"].apply(str)
    # easy_problems["problem_instance"] = (
    #     easy_problems["param"] + easy_problems["instance"]
    # )
    # easy_problems = easy_problems["problem_instance"]
    # run_over_files(results_dir="results/paper_version_TTPN_10_min", timeout=18000)
    # # print(solve_small_problem_math())
    print(
        solve_file_problem(
            path="extract_problems/data/j60.sm.tgz/j6040_2.sm",
            # timed_transition=True,
            logging=True,
        )
    )

    # print(
    #     solve_file_problem(
    #         path="extract_problems/data/j30.sm.tgz/j3014_5.sm",
    #         timed_transition=True,
    #         logging=True,
    #     )
    # )
    # solve_rcpsp_optimizer(path="extract_problems/data/j30.sm.tgz/j3014_5.sm")


if __name__ == "__main__":
    main()
