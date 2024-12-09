import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter

from real_projects.real_projects_petri_modeling import (
    analyze_path,
    get_activity_metadata,
    create_augmented_petri_net,
    find_split_points,
    astar_search,
)


def csv_to_xes(csv_path):
    data = pd.read_csv(csv_path)
    cols = ["case:concept:name", "concept:name", "time:timestamp"]
    data = data[cols]
    data["time:timestamp"] = pd.to_datetime(data["time:timestamp"])
    log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    pm4py.write_xes(
        log, csv_path.replace(".csv", ".xes"), case_id_key="case:concept:name"
    )


def smaller_data(file_to_handle):
    full_data = pd.read_csv(file_to_handle)
    cols = ["case:concept:name", "concept:name", "time:timestamp", "Duration"]
    full_data["concept:name"] = full_data["concept:name"].apply(str)
    full_data = full_data[cols]
    full_data["resource_demand_R1"] = np.random.randint(1, 6, size=len(full_data))
    full_data["resource_demand_R2"] = np.random.randint(4, 8, size=len(full_data))
    # full_data.loc[full_data["concept:name"] == "j", "resource_demand_R1"] = 2
    # small_data = full_data[(full_data["case:concept:name"] != 2)]
    # small_data = full_data
    small_data = full_data[
        (full_data["concept:name"].apply(len) < 2) & (full_data["concept:name"] < "v")
    ]
    output_path = f"small_{file_to_handle}"
    small_data.to_csv(output_path, index=False)
    return output_path


def analyze_csv_path(csv_path, available_resources=None):
    csv_to_xes(csv_path)
    log_3 = pm4py.read_xes(csv_path.replace(".csv", ".xes"))
    net, im, fm = pm4py.discover_petri_net_inductive(log_3)

    # Get metadata and create augmented net
    metadata = get_activity_metadata(csv_path)
    view_net(fm, im, net, metadata)
    split_points = find_split_points(net, 3)
    print(split_points)
    available_resources = {"R1": 12, "R2": 15}
    aug_net, initial_marking = create_augmented_petri_net(
        net, im, metadata, available_resources
    )

    all_transitions = set(net.transitions)

    # Run search
    path = astar_search(
        aug_net=aug_net,
        initial_marking=initial_marking,
        goal_marking=fm,
        available_resources=available_resources,
        debug=True,
    )

    analyze_path(path, aug_net, all_transitions)


def view_net(fm, im, net, transition_times):
    # Set transition labels if they exist
    for transition in net.transitions:
        if transition.label:
            transition.name = (
                f"{transition.label}\nt={transition_times[transition.label]['duration']}"
                f"\nR₁={transition_times[transition.label]['resources']['R1']}"
            )

    # Create decorations dictionary for transitions
    # Set up visualization parameters
    parameters = {"format": "png", "debug": True}

    # Generate visualization
    from pm4py.visualization.petri_net import visualizer

    gviz = visualizer.apply(net, im, fm, parameters=parameters)

    # View the visualization
    # visualizer.view(gviz)
    visualizer.save(gviz, "example_petri_net_with_resources.png")


output_path = smaller_data("house_Processed.csv")
analyze_csv_path(output_path)
# smaller_data()
