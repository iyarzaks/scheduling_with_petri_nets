import pygraphviz as pgv

from RCPSP_modeling.rcpsp_base import RcpspBase

START = "_start"
FINISH = "_finish"
PRE = "_pre_"
POST = "post_"
PLACE = "place"
COUNT = "count"
TIME = "time"


def merge_dicts(dict1, dict2, dict3=None):
    dict2.update(dict1)
    if dict3:
        dict3.update(dict2)
        return dict3
    return dict2


class PlaceTimePetriNetTransition:
    def __init__(self, name, arcs_in, arcs_out):
        self.name = name
        self.arcs_in = arcs_in
        self.arcs_out = arcs_out
        self.net = pgv.AGraph(directed=True)

    def is_available(self, marking):
        return all(
            marking[place].get(COUNT, 0) >= self.arcs_in[place]
            for place in self.arcs_in
        )


class PlaceTimePetriNetPlace:
    def __init__(self, name, arcs_in, arcs_out, duration=0, state=None):
        self.name = name
        self.arcs_in = arcs_in
        self.arcs_out = arcs_out
        if state is None:
            self.state = dict()
        else:
            self.state = state
        self.duration = duration


class RcpspPlaceTimePetriNet:
    def __init__(self, rcpsp_basic: RcpspBase):
        self.transitions_dict = {}
        self.places_dict = {}
        self.net = pgv.AGraph(directed=True)
        self.transitions = []
        self.places = []
        # for nuw we assume that the activities sorted in a way that if for every i,
        # j if i<j j nod depend on i in any case
        for resource in rcpsp_basic.resources:
            self.places.append(
                PlaceTimePetriNetPlace(
                    name=resource,
                    arcs_in={
                        activity.name + FINISH: activity.resource_demands[resource]
                        for activity in rcpsp_basic.activities
                        if resource in activity.resource_demands.keys()
                    },
                    arcs_out={
                        activity.name + START: activity.resource_demands[resource]
                        for activity in rcpsp_basic.activities
                        if resource in activity.resource_demands.keys()
                    },
                    state={COUNT: rcpsp_basic.resources[resource], TIME: 0},
                )
            )
        for activity in rcpsp_basic.activities:
            self.add_activity(activity, rcpsp_basic)
        self.update_net()

    def add_activity(self, activity, rcpsp_basic):
        if activity.name not in rcpsp_basic.backward_dependencies:
            # no dependent activity add start place
            self.places.append(
                PlaceTimePetriNetPlace(
                    name=PRE + activity.name,
                    arcs_in=dict(),
                    arcs_out={activity.name + START: 1},
                    duration=0,
                    state={COUNT: 1, TIME: 0},
                )
            )
        self.places.append(
            # add activity place
            PlaceTimePetriNetPlace(
                name=activity.name,
                arcs_in={
                    activity.name + START: sum(activity.resource_demands.values()) + 1
                },
                arcs_out={
                    activity.name + FINISH: sum(activity.resource_demands.values()) + 1
                },
                duration=activity.duration,
            )
        )
        self.transitions.append(
            # add start of activity transition
            PlaceTimePetriNetTransition(
                name=activity.name + START,
                arcs_in=merge_dicts(
                    {
                        place.name: 1
                        for place in self.places
                        if activity.name + START in place.arcs_out
                        and place.name not in rcpsp_basic.resources
                    },
                    {
                        resource: activity.resource_demands[resource]
                        for resource in activity.resource_demands.keys()
                    },
                ),
                arcs_out={activity.name: sum(activity.resource_demands.values()) + 1},
            )
        )
        post_no_dependencies_dict = (
            {POST + activity.name: 1}
            if activity.name not in rcpsp_basic.dependencies
            or len(rcpsp_basic.dependencies[activity.name]) == 0
            else {}
        )
        self.transitions.append(
            # add end of activity transition
            PlaceTimePetriNetTransition(
                name=activity.name + FINISH,
                arcs_in={activity.name: sum(activity.resource_demands.values()) + 1},
                arcs_out=merge_dicts(
                    {
                        POST + activity.name + PRE + activity_successor: 1
                        for activity_successor in rcpsp_basic.dependencies.get(
                            activity.name, []
                        )
                    },
                    post_no_dependencies_dict,
                    {
                        resource: activity.resource_demands[resource]
                        for resource in activity.resource_demands.keys()
                    },
                ),
            )
        )
        for successor_activity in rcpsp_basic.dependencies.get(activity.name, []):
            self.places.append(
                PlaceTimePetriNetPlace(
                    name=POST + activity.name + PRE + successor_activity,
                    arcs_in={activity.name + FINISH: 1},
                    arcs_out={successor_activity + START: 1},
                )
            )
        if (
            activity.name not in rcpsp_basic.dependencies
            or len(rcpsp_basic.dependencies[activity.name]) == 0
        ):
            self.places.append(
                PlaceTimePetriNetPlace(
                    name=POST + activity.name,
                    arcs_in={activity.name + FINISH: 1},
                    arcs_out=dict(),
                )
            )

    def update_net(self):
        for place in self.places:
            self.net.add_node(
                place.name,
                shape="circle",
                color="lightblue",
                label=f"{place.name}\n{place.state}",
            )
            for successor in place.arcs_in:
                self.net.add_edge(successor, place.name, label=place.arcs_in[successor])
            self.places_dict[place.name] = place
        for transition in self.transitions:
            self.net.add_node(transition.name, shape="box", color="lightgreen")
            for successor in transition.arcs_in:
                self.net.add_edge(
                    successor, transition.name, label=transition.arcs_in[successor]
                )
            self.transitions_dict[transition.name] = transition

    def plot(self, filename):
        self.net.layout(prog="dot")
        self.net.draw(filename)
