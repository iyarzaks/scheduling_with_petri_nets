from RCPSP_modeling.rcpsp_base import RcpspBase
from RCPSP_modeling.rcpsp_petri_net import RcpspPlaceTimePetriNet


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
    petri_example = RcpspPlaceTimePetriNet(rcpsp_example)
    petri_example.plot("test.png")
    print(petri_example)
