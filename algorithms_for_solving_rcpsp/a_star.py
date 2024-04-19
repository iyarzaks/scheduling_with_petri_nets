from RCPSP_modeling.rcpsp_petri_net import PlaceTimePetriNetTransition, TIME, COUNT
from RCPSP_modeling.rcpsp_petri_net import RcpspPlaceTimePetriNet


class RgNode:
    def __init__(
        self,
        petri_net_to_solve,
        previous_rg_node=None,
        transition: PlaceTimePetriNetTransition = None,
    ):
        self.marking = {}
        if previous_rg_node is None:
            for place in petri_net_to_solve.places:
                if place.state is not None:
                    self.marking[place.name] = place.state
        else:
            self.marking = self.update_marking(
                previous_rg_node.marking, transition, petri_net_to_solve
            )
        self.available_transitions = [
            transition
            for transition in petri_net_to_solve.transitions
            if transition.is_available(self.marking)
        ]

    def update_marking(
        self,
        previous_marking,
        transition: PlaceTimePetriNetTransition,
        petri_net_to_solve,
    ):
        new_marking = previous_marking
        for out_node in transition.arcs_out:
            new_marking[out_node] = {
                TIME: previous_marking[out_node][TIME]
                + petri_net_to_solve.activities[out_node].duration,
                COUNT: transition.arcs_out[out_node],
            }
        return new_marking


class AStarSolver:
    def __init__(self, petri_net_to_solve: RcpspPlaceTimePetriNet, heuristic_function):
        self.petri_net_to_solve = petri_net_to_solve
        self.heuristic_function = heuristic_function
        self.start_node = RgNode(petri_net_to_solve)
        print(self.start_node)

    def heuristic(self, current_rg_node):

        pass

    def solve(self):
        pass
