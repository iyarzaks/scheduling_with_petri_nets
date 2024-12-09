from RCPSP_modeling.rcpsp_base import RcpspBase
from RCPSP_modeling.rcpsp_petri_net import (
    RcpspTimedPlacePetriNet,
    RcpspTimedTransitionPetriNet,
)
from extract_problems.extract_problem import extract_rcpsp


def test_modeling():
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
    petri_example = RcpspTimedPlacePetriNet(rcpsp_example)
    petri_example.plot("test.png")


def test_modeling_timed_transition():
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
    petri_example = RcpspTimedTransitionPetriNet(rcpsp_example)
    petri_example.plot("test_timed_transition.png")
    print(petri_example)


def test_modeling_j_30():
    rcpsp_example = extract_rcpsp(
        "/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j3026_4.txt"
    )
    petri_example = RcpspTimedPlacePetriNet(rcpsp_example)
    petri_example.plot("test_30j.png")
    print(petri_example)
