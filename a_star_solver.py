from RCPSP_modeling.rcpsp_base import RcpspBase
from RCPSP_modeling.rcpsp_petri_net import RcpspPlaceTimePetriNet
from algorithms_for_solving_rcpsp.a_star import AStarSolver
from extract_problems.extract_problem import extract_rcpsp


def cp_heuristic(rcpsp_example: RcpspBase, removed_activities):
    rcpsp_example.update_problem(removed_activities=removed_activities)
    return rcpsp_example.calculate_critical_path()["duration"]


def solver():
    petri_example, rcpsp_example = init_real_30_example()
    a_star_solver = AStarSolver(
        petri_example, rcpsp_example, heuristic_function=cp_heuristic
    )
    a_star_solver.solve()


def init_real_30_example():
    rcpsp_example = extract_rcpsp(
        "/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j3026_4.txt"
    )
    petri_example = RcpspPlaceTimePetriNet(rcpsp_example)
    return petri_example, rcpsp_example


def init_small_example():
    rcpsp_example = RcpspBase(
        activities_list=[
            {"name": "A", "duration": 8, "resource_demands": {"R1": 4}},
            {"name": "B", "duration": 4, "resource_demands": {"R1": 4}},
            {"name": "C", "duration": 4, "resource_demands": {"R1": 3}},
            {"name": "D", "duration": 3, "resource_demands": {"R1": 5}},
        ],
        dependencies={"A": ["C", "D"], "B": ["D"]},
        resources={"R1": 12},
    )
    petri_example = RcpspPlaceTimePetriNet(rcpsp_example)
    return petri_example, rcpsp_example


def main():
    solver()


if __name__ == "__main__":
    main()
