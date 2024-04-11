from RCPSP_modeling.rcpsp_base import RcpspBase
from RCPSP_modeling.rcpsp_petri_net import RcpspPlaceTimePetriNet
from algorithms_for_solving_rcpsp.a_star import AStarSolver


def dummy_heuristic():
    return 0


def test_solver():
    petri_example = init_example()
    a_star_solver = AStarSolver(petri_example, heuristic_function=dummy_heuristic)


def init_example():
    rcpsp_example = RcpspBase(
        activities_list=[
            {"name": "A", "duration": 8, "resource_demands": {"R1": 4}},
            {"name": "B", "duration": 4, "resource_demands": {"R1": 10}},
            {"name": "C", "duration": 4, "resource_demands": {"R1": 3}},
            {"name": "D", "duration": 3, "resource_demands": {"R1": 5}},
        ],
        dependencies={"A": ["C", "D"], "B": ["D"]},
        resources={"R1": 12},
    )
    petri_example = RcpspPlaceTimePetriNet(rcpsp_example)
    return petri_example
