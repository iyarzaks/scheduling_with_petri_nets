import pygraphviz as pgv

from RCPSP_modeling.rcpsp_base import RcpspBase

START = "_start"
FINISH = "_finish"
PRE = "_pre_"
POST = "post_"
PLACE = "place"


class PlaceTimePetriNetTransition:
    def __init__(self, name, arcs_in, arcs_out):
        self.name = name
        self.arcs_in = arcs_in
        self.arcs_out = arcs_out
        self.net = pgv.AGraph(directed=True)


class PlaceTimePetriNetPlace:
    def __init__(self, name, arcs_in, arcs_out, duration=0, state=None):
        self.name = name
        self.arcs_in = arcs_in
        self.arcs_out = arcs_out
        if state is None:
            self.state = {0: 0}
        else:
            self.state = state
        self.duration = duration


class RcpspPlaceTimePetriNet:
    def __init__(self, rcpsp_basic: RcpspBase):
        self.net = pgv.AGraph(directed=True)
        self.transitions = []
        self.places = []
        # for nuw we assume that the activities sorted in a way that if for every i,
        # j if i<j j nod depend on i in any case
        for resource in rcpsp_basic.resources:
            self.places.append(
                PlaceTimePetriNetPlace(
                    name=resource,
                    arcs_in=[
                        activity.name + FINISH
                        for activity in rcpsp_basic.activities
                        if resource in activity.resource_demands.keys()
                    ],
                    arcs_out=[
                        activity.name + START
                        for activity in rcpsp_basic.activities
                        if resource in activity.resource_demands.keys()
                    ],
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
                    arcs_in=[],
                    arcs_out=[activity.name + START],
                    duration=0,
                    state={1: 0},
                )
            )
        self.places.append(
            # add activity place
            PlaceTimePetriNetPlace(
                name=activity.name,
                arcs_in=[activity.name + START],
                arcs_out=[activity.name + FINISH],
                duration=activity.duration,
            )
        )
        self.transitions.append(
            # add start of activity transition
            PlaceTimePetriNetTransition(
                name=activity.name + START,
                arcs_in=[
                    place.name
                    for place in self.places
                    if activity.name + START in place.arcs_out
                ]
                + [resource for resource in activity.resource_demands.keys()],
                arcs_out=[activity.name],
            )
        )
        self.transitions.append(
            # add end of activity transition
            PlaceTimePetriNetTransition(
                name=activity.name + FINISH,
                arcs_in=[activity.name],
                arcs_out=[
                    POST + activity.name + PRE + activity_successor
                    for activity_successor in rcpsp_basic.dependencies.get(
                        activity.name, []
                    )
                ]
                + [resource for resource in activity.resource_demands.keys()],
            )
        )
        for successor_activity in rcpsp_basic.dependencies.get(activity.name, []):
            self.places.append(
                PlaceTimePetriNetPlace(
                    name=POST + activity.name + PRE + successor_activity,
                    arcs_in=[activity.name + FINISH],
                    arcs_out=[successor_activity + START],
                )
            )
        if activity.name not in rcpsp_basic.dependencies:
            self.places.append(
                PlaceTimePetriNetPlace(
                    name=POST + activity.name,
                    arcs_in=[activity.name + FINISH],
                    arcs_out=[],
                )
            )

    def update_net(self):
        for place in self.places:
            self.net.add_node(
                place.name, shape="circle", color="lightblue", width=0.01, height=0.01
            )
            for successor in place.arcs_in:
                self.net.add_edge(successor, place.name)
        for transition in self.transitions:
            self.net.add_node(
                transition.name,
                shape="box",
                color="lightgreen",
                width=0.01,
                height=0.01,
            )
            for successor in transition.arcs_in:
                self.net.add_edge(successor, transition.name)

    def plot(self, filename):
        self.net.layout(prog="dot")
        self.net.draw(filename)
