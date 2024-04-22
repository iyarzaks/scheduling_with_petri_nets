from RCPSP_modeling.rcpsp_base import RcpspBase


def test_critical_path():
    rcpsp_example = RcpspBase(
        activities_list=[
            {"name": "A", "duration": 2, "resource_demands": {"R1": 4}},
            {"name": "B", "duration": 3, "resource_demands": {"R1": 10}},
            {"name": "C", "duration": 4, "resource_demands": {"R1": 3}},
            {"name": "D", "duration": 6, "resource_demands": {"R1": 5}},
        ],
        dependencies={"A": ["C"], "C": ["D"], "B": ["C"]},
        resources={"R1": 12},
    )
    print("\n")
    res = rcpsp_example.calculate_critical_path()
    print(res)
    rcpsp_example.update_problem(removed_activities=["B", "A"])
    res = rcpsp_example.calculate_critical_path()
    print(res)
