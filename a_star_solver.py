from RCPSP_modeling.rcpsp_base import RcpspBase
from RCPSP_modeling.rcpsp_petri_net import RcpspPlaceTimePetriNet
from algorithms_for_solving_rcpsp.a_star import AStarSolver
from extract_problems.extract_problem import extract_rcpsp


def cp_heuristic(
    rcpsp_example: RcpspBase, removed_activities, started_activities, marking
):
    rcpsp_example.update_problem(removed_activities=started_activities)
    return rcpsp_example.calculate_critical_path()["duration"]


def solver():
    petri_example, rcpsp_example = init_real_30_example()
    a_star_solver = AStarSolver(
        petri_example, rcpsp_example, heuristic_function=cp_heuristic
    )
    a_star_solver.solve(beam_search_size=10)


def init_real_30_example():
    rcpsp_example = extract_rcpsp(
        "/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j60.sm.tgz/j6038_10.sm"
    )
    petri_example = RcpspPlaceTimePetriNet(rcpsp_example)
    return petri_example, rcpsp_example


def init_small_example():
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
    petri_example = RcpspPlaceTimePetriNet(rcpsp_example)
    return petri_example, rcpsp_example


def main():
    solver()


if __name__ == "__main__":
    main()
