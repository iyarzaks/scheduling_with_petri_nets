import copy
from collections import deque
from heapq import heapify, heappush, heappop

import numpy as np


class Place:
    def __init__(self, name, in_arcs=None, out_arcs=None, properties={}):
        self.name = name
        self.in_arcs = set() if in_arcs is None else in_arcs
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.properties = properties

    def __repr__(self):
        return self.name


class Transition:
    def __init__(
        self,
        name,
        label,
        in_arcs=None,
        out_arcs=None,
        move_type=None,
        prob=None,
        weight=None,
        properties={},
        cost_function=None,
    ):
        self.name = name
        self.label = label
        self.in_arcs = set() if in_arcs is None else in_arcs
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.move_type = move_type
        self.prob = prob
        self.cost_function = cost_function
        self.weight = self.__initialize_weight(weight)
        self.properties = properties

    def __repr__(self):
        return self.name

    def __initialize_weight(self, weight):
        if weight is not None:
            return weight

        if self.prob == 0:
            return np.inf

        if self.cost_function is None:
            return 0 if self.move_type == "sync" else 1
            # raise Exception("A cost function should be defined for transitions if no weight is given and prob is positive")
        return self.cost_function(self.prob)


class Arc:
    def __init__(self, source, target, weight=1, properties={}):
        self.source = source
        self.target = target
        self.weight = weight
        self.properties = properties

    def __repr__(self):
        return self.source.name + " -> " + self.target.name


class Marking:
    def __init__(self, places=None):
        self.places = tuple(0 for place in places) if places is None else places

    def __repr__(self):
        return str(self.places)


class Node:
    def __init__(self, marking):
        self.marking = marking
        self.neighbors = set()

    def __repr__(self):
        return str(self.marking)

    def add_neighbor(self, node, transition):
        self.neighbors.add((node, transition))


class Edge:
    def __init__(self, name, source_marking, target_marking, move_type):
        self.name = name
        self.source_marking = source_marking
        self.target_marking = target_marking
        self.move_type = move_type

    def __repr__(self):
        return f"{self.source_marking} -> {self.name} -> {self.target_marking}"


class Graph:
    def __init__(self, nodes=None, edges=None, starting_node=None, ending_node=None):
        self.nodes = list() if nodes is None else nodes
        self.edges = list() if edges is None else edges
        self.starting_node = starting_node
        self.ending_node = ending_node
        self.nodes_indices = {}

    def __repr__(self):
        return f"Nodes:{self.nodes}, \n edges:{self.edges}"

    def __get_markings(self):
        return set([node.marking for node in self.nodes])

    def add_node(self, node):
        self.nodes.append(node)
        self.nodes_indices[node.marking] = len(self.nodes) - 1

    def add_edge(self, edge):
        self.edges.append(edge)


# --- Original ---
# class search_node_new:
#     def __init__(self, marking, dist=np.inf, ancestor=None, transition_to_ancestor=None, path_prefix=''):
#         self.dist = dist
#         self.ancestor = ancestor
#         self.transition_to_ancestor = transition_to_ancestor
#         self.marking = marking
#         self.path_prefix = path_prefix

#     def __lt__(self, other):
#         return self.dist <= other.dist

#     def __repr__(self):
#         return f'Node: {self.graph_node.marking}, dist:{self.dist}'


# Extended to compare transition label
class search_node_new:
    def __init__(
        self,
        marking,
        dist=np.inf,
        ancestor=None,
        transition_to_ancestor=None,
        path_prefix="",
    ):
        self.dist = dist
        self.ancestor = ancestor
        self.transition_to_ancestor = transition_to_ancestor
        self.marking = marking
        self.path_prefix = path_prefix

    def __lt__(self, other):
        if self.dist == other.dist:
            self_label = getattr(self.transition_to_ancestor, "label", None)
            other_label = getattr(other.transition_to_ancestor, "label", None)
            return self_label is not None and other_label is None
        return self.dist < other.dist

    def __repr__(self):
        return f"Node: {self.marking}, dist: {self.dist}"


class PetriNet:
    def __init__(
        self,
        name="net",
        places=None,
        transitions=None,
        arcs=None,
        properties={},
        conditioned_prob_compute=False,
    ):
        self.name = name
        self.transitions = list() if transitions is None else transitions
        self.places = list() if places is None else places
        self.arcs = list() if arcs is None else arcs
        self.properties = properties
        self.init_mark = None
        self.final_mark = None
        self.reachability_graph = None
        self.places_indices = {self.places[i].name: i for i in range(len(self.places))}
        self.transitions_indices = {
            self.transitions[i].name: i for i in range(len(self.transitions))
        }
        self.cost_function = None
        self.conditioned_prob_compute = conditioned_prob_compute

    def construct_reachability_graph(self):
        curr_mark = self.init_mark
        curr_node = Node(curr_mark.places)
        self.reachability_graph = Graph()
        if self.final_mark is not None:
            self.reachability_graph.ending_node = Node(self.final_mark.places)
        self.reachability_graph.add_node(curr_node)
        self.reachability_graph.starting_node = curr_node
        available_transitions = self._find_available_transitions(curr_mark.places)
        nodes_to_explore = deque()
        visited_marks = set()

        for transition in available_transitions:
            nodes_to_explore.append((curr_mark, transition, curr_node))

        visited_marks.add(curr_mark.places)

        while nodes_to_explore:
            prev_node_triplet = nodes_to_explore.popleft()
            prev_mark, prev_transition, prev_node = (
                prev_node_triplet[0],
                prev_node_triplet[1],
                prev_node_triplet[2],
            )
            assert (
                self.__check_transition_prerequesits(prev_transition, prev_mark.places)
                == True
            )
            curr_mark = self._fire_transition(prev_mark, prev_transition)

            if curr_mark.places in visited_marks:
                node_idx = self.reachability_graph.nodes_indices[curr_mark.places]
                curr_node = self.reachability_graph.nodes[node_idx]
            else:
                curr_node = Node(curr_mark.places)

            prev_node.add_neighbor(curr_node, prev_transition)
            self.reachability_graph.add_edge(
                Edge(
                    prev_transition.name,
                    prev_mark,
                    curr_mark,
                    prev_transition.move_type,
                )
            )

            if curr_mark.places in visited_marks:
                continue

            else:
                for transition in self._find_available_transitions(curr_mark.places):
                    nodes_to_explore.append((curr_mark, transition, curr_node))

                visited_marks.add(curr_mark.places)
                self.reachability_graph.add_node(curr_node)

    def construct_synchronous_product(self, trace_model, cost_function):
        """This func assigns all trace transitions move_type=trace and all model transitions move_type=model
        additionaly, all sync transitions will be assigned move_type=sync"""

        self.assign_model_transitions_move_type()
        trace_model.assign_trace_transitions_move_type()
        sync_places = copy.copy(self.places + trace_model.places)
        sync_transitions = copy.copy(self.transitions + trace_model.transitions)
        sync_arcs = copy.copy(self.arcs + trace_model.arcs)

        new_sync_transitions = self._generate_all_sync_transitions(
            trace_model, cost_function
        )
        sync_prod = PetriNet("sync_prod", sync_places, sync_transitions, sync_arcs)

        sync_prod.add_transitions_with_arcs(new_sync_transitions)
        sync_prod.init_mark = Marking(
            self.init_mark.places + trace_model.init_mark.places
        )
        sync_prod.final_mark = Marking(
            self.final_mark.places + trace_model.final_mark.places
        )
        self._update_sync_product_trans_names(sync_prod)
        print("Wrong function Dude!! -- def construct_synchronous_product")
        return sync_prod

    def add_places(self, places):
        if isinstance(places, list):
            self.places += places

        else:
            self.places.append(places)

        self.__update_indices_p_dict(places)

    def add_transitions(self, transitions):
        if isinstance(transitions, list):
            self.transitions += transitions

        else:
            self.transitions.append(transitions)

        self.__update_indices_t_dict(transitions)

    def add_transitions_with_arcs(self, transitions):
        if isinstance(transitions, list):
            self.transitions += transitions
            for transition in transitions:
                self.arcs += list(transition.in_arcs.union(transition.out_arcs))

        else:
            self.transitions.append(transitions)
            self.arcs += list(transition.in_arcs.union(transition.out_arcs))

        self.__update_indices_t_dict(transitions)

    def add_arc_from_to(self, source, target, weight=None):
        if weight is None:
            arc = Arc(source, target)
        else:
            arc = Arc(source, target, weight)
        source.out_arcs.add(arc)
        target.in_arcs.add(arc)
        self.arcs.append(arc)

    def _generate_all_sync_transitions(self, trace_model, cost_function):
        sync_transitions = []
        counter = 1

        for trans in self.transitions:
            # trans.label is guaranteed to be unique in the discovered model (from docs)
            if trans.label is not None:
                # Find in the trace model all the transitions with the same label
                same_label_transitions = self.__find_simillar_label_transitions(
                    trace_model, trans.label
                )

                for trace_trans in same_label_transitions:
                    new_sync_trans = self.__generate_new_trans(
                        trans, trace_trans, counter, cost_function
                    )
                    sync_transitions.append(new_sync_trans)
                    counter += 1

        return sync_transitions

    def __find_simillar_label_transitions(self, trace_model, activity_label):
        """Returns all the transitions in the trace with a specified activity label"""
        same_label_trans = [
            transition
            for transition in trace_model.transitions
            if transition.label == activity_label
        ]

        return same_label_trans

    def __generate_new_trans(self, trans, trace_trans, counter, cost_function):
        #         name = 'sync_transition_' + str(counter)
        name = f"sync_{trans.name}"
        new_sync_transition = Transition(
            name=name,
            label=trans.label,
            move_type="sync",
            prob=trace_trans.prob,
            cost_function=cost_function,
        )

        input_arcs = trans.in_arcs.union(trace_trans.in_arcs)
        new_input_arcs = []
        for arc in input_arcs:
            new_arc = Arc(arc.source, new_sync_transition, arc.weight)
            new_input_arcs.append(new_arc)

        output_arcs = trans.out_arcs.union(trace_trans.out_arcs)
        new_output_arcs = []
        for arc in output_arcs:
            new_arc = Arc(new_sync_transition, arc.target, arc.weight)
            new_output_arcs.append(new_arc)

        new_sync_transition.in_arcs = new_sync_transition.in_arcs.union(new_input_arcs)
        new_sync_transition.out_arcs = new_sync_transition.out_arcs.union(
            new_output_arcs
        )

        return new_sync_transition

    def __update_indices_p_dict(self, places):
        curr_idx = len(self.places_indices)
        if isinstance(places, list):
            for p in places:
                self.places_indices[p.name] = curr_idx
                curr_idx += 1
        else:
            self.places_indices[places.name] = curr_idx

    def __update_indices_t_dict(self, transitions):
        curr_idx = len(self.transitions_indices)
        if isinstance(transitions, list):
            for t in transitions:
                self.transitions_indices[t.name] = curr_idx
                curr_idx += 1
        else:
            self.transitions_indices[transitions.name] = curr_idx

    def _find_available_transitions(self, mark_tuple):
        """Input: tuple
        Output: list"""

        available_transitions = []
        for transition in self.transitions:
            if self.__check_transition_prerequesits(transition, mark_tuple):
                available_transitions.append(transition)

        return available_transitions

    def __check_transition_prerequesits(self, transition, mark_tuple):
        for arc in transition.in_arcs:
            arc_weight = arc.weight
            source_idx = self.places_indices[arc.source.name]
            if mark_tuple[source_idx] < arc_weight:
                return False

        return True

    def __assign_trace_transitions_move_type(self):
        for trans in self.transitions:
            if trans.move_type is None:
                trans.move_type = "trace"

    def assign_trace_transitions_move_type(self):
        return self.__assign_trace_transitions_move_type()

    def assign_model_transitions_move_type(self):
        return self.__assign_model_transitions_move_type()

    def __assign_model_transitions_move_type(self):
        for trans in self.transitions:
            if trans.move_type is None:
                trans.move_type = "model"

    def conformance_checking(self, trace_model, hist_prob_dict=None, lamda=0.5):
        sync_prod = self.construct_synchronous_product(trace_model, self.cost_function)
        return sync_prod._dijkstra_no_rg_construct(hist_prob_dict, lamda=lamda)

    def __dijkstra(self):
        distance_min_heap = []
        heapify(distance_min_heap)
        #         visited_nodes = set()
        search_graph_nodes = [
            search_node(node) for node in self.reachability_graph.nodes
        ]
        nodes_idx_dict = {
            search_node.graph_node.marking: idx
            for idx, search_node in enumerate(search_graph_nodes)
        }

        source_node_idx = nodes_idx_dict[self.reachability_graph.starting_node.marking]
        source_node = search_graph_nodes[source_node_idx]
        source_node.dist = 0

        for node in search_graph_nodes:
            heappush(distance_min_heap, node)

        while distance_min_heap:
            min_dist_node = heappop(distance_min_heap)
            need_heapify = False

            for neighbor_transition_tuple in min_dist_node.graph_node.neighbors:
                neighbor, transition = (
                    neighbor_transition_tuple[0],
                    neighbor_transition_tuple[1],
                )
                alt_distance = min_dist_node.dist + transition.weight
                neighbor_search_idx = nodes_idx_dict[neighbor.marking]

                if alt_distance < search_graph_nodes[neighbor_search_idx].dist:
                    search_graph_nodes[neighbor_search_idx].dist = alt_distance
                    search_graph_nodes[neighbor_search_idx].ancestor = min_dist_node
                    search_graph_nodes[neighbor_search_idx].transition_to_ancestor = (
                        transition
                    )
                    need_heapify = True

            if need_heapify:
                heapify(distance_min_heap)

        #         print('ending marking is: ', self.reachability_graph.ending_node.marking)
        #         print('nodes_idx_dict is: ', nodes_idx_dict)
        final_mark_idx = nodes_idx_dict[self.reachability_graph.ending_node.marking]
        curr_node = search_graph_nodes[final_mark_idx]
        path = []

        while curr_node is not source_node:
            #             path.append(curr_node.transition_to_ancestor.label)
            path.append(curr_node.transition_to_ancestor.name)
            curr_node = curr_node.ancestor

        #         print(f'Shortest path len: {search_graph_nodes[final_mark_idx].dist}, \n Optimal alignment: {path[::-1]}')
        return path[::-1], search_graph_nodes[final_mark_idx].dist

    def _dijkstra_no_rg_construct(
        self, prob_dict, lamda=0.5, return_final_marking=False
    ):
        distance_min_heap = []
        heapify(distance_min_heap)
        curr_node = search_node_new(self.init_mark, dist=0)
        heappush(distance_min_heap, curr_node)
        marking_distance_dict = {}
        visited_markings = set()

        if prob_dict is None:
            prob_dict = {}

        while distance_min_heap:
            min_dist_node = heappop(distance_min_heap)

            if min_dist_node.marking.places in visited_markings:
                continue

            if min_dist_node.marking.places == self.final_mark.places:
                break

            available_transitions = self._find_available_transitions(
                min_dist_node.marking.places
            )
            for transition in available_transitions:
                new_marking = self._fire_transition(min_dist_node.marking, transition)

                if new_marking.places in visited_markings:
                    continue

                if new_marking in visited_markings:
                    continue

                conditioned_transition_weight = self.compute_conditioned_weight(
                    min_dist_node.path_prefix, transition, prob_dict, lamda=lamda
                )
                if (
                    new_marking.places not in marking_distance_dict
                    or marking_distance_dict[new_marking.places]
                    > min_dist_node.dist + conditioned_transition_weight
                ):
                    new_path_prefix = (
                        min_dist_node.path_prefix + transition.label
                        if transition.label is not None
                        else min_dist_node.path_prefix
                    )

                    new_node = search_node_new(
                        new_marking,
                        dist=min_dist_node.dist + conditioned_transition_weight,
                        ancestor=min_dist_node,
                        transition_to_ancestor=transition,
                        path_prefix=new_path_prefix,
                    )

                    marking_distance_dict[new_marking.places] = new_node.dist
                    heappush(distance_min_heap, new_node)

            visited_markings.add(min_dist_node.marking.places)

        shortest_path = []
        curr_node = min_dist_node
        while curr_node.ancestor:
            shortest_path.append(curr_node.transition_to_ancestor.name)
            curr_node = curr_node.ancestor

        #         print('min_dist_node distance: ', min_dist_node.dist)
        #         print('shortest path: ', shortest_path[::-1])
        if return_final_marking:  # TO DO: need to include overlap in the code
            return shortest_path[::-1], min_dist_node.dist, self.marking.place

        return shortest_path[::-1], min_dist_node.dist

    def astar_extended(self, k=None, s=0):
        if k is None:
            k = set()

        visited_markings = set()
        visited_markings_distance_dict = {}
        distance_min_heap = []
        heapify(distance_min_heap)

        # these two should be new attributes in the petri net instead of copying them each time
        incidence_mat = (
            self.compute_incidence_matrix()
        )  # Need to check the condition (t,p) not in F
        consump_mat = self.compute_consumption_matrix()

        init_node = self.initialize_min_dist_node(k, incidence_mat, consump_mat)
        heappush(distance_min_heap, init_node)
        visited_markings_distance_dict[self.init_mark] = init_node
        final_min_dist = np.inf
        node = None

        while distance_min_heap:
            need_heapify = False
            min_dist_node = heappop(distance_min_heap)

            if min_dist_node.marking == self.final_mark:
                break

            if min_dist_node.have_estimated_solution:
                max_events_explained = max(min_dist_node.n_explained_events, s)
                if max_events_explained not in k:
                    k.add(max_events_explained)
                    return self.astar_extended(k, s=0)

                heuristic_distance, sol_vec = self.compute_heuristic_extended(
                    None,
                    self.transitions_weights,
                    np.array(min_dist_node.marking.places),
                    np.array(self.final_mark.places),
                    incidence_mat,
                    consump_mat,
                    init_node_comp=False,
                    n_explained_events=min_dist_node.n_explained_events,
                )

                min_dist_node.disappointing = True

                if sol_vec is not None:
                    min_dist_node.have_exact_known_solution = True
                    min_dist_node.have_estimated_solution = False
                    min_dist_node.solution_vec = sol_vec

                else:
                    min_dist_node.heuristic_distance = np.inf
                    min_dist_node.total_distance = (
                        min_dist_node.dist_from_origin
                        + min_dist_node.heuristic_distance
                    )
                    min_dist_node.solution_vec = None
                    heappush(distance_min_heap, min_dist_node)
                    continue

                if heuristic_distance > min_dist_node.heuristic_distance:
                    min_dist_node.heuristic_distance = heuristic_distance
                    min_dist_node.total_distance = (
                        min_dist_node.dist_from_origin
                        + min_dist_node.heuristic_distance
                    )
                    heappush(distance_min_heap, min_dist_node)
                    continue

            s = max(s, min_dist_node.n_explained_events)
            visited_markings.add(min_dist_node.marking)

            for transition in self.__find_available_transitions(
                min_dist_node.marking.places
            ):
                new_mark = self.__fire_transition(min_dist_node.marking, transition)
                need_push_node = False
                transition_idx = self.transitions_indices[transition.name]

                if new_mark not in visited_markings:
                    dist_to_node = min_dist_node.dist_from_origin + transition.weight
                    sol_vec = np.array(min_dist_node.solution_vec)

                    if new_mark in visited_markings_distance_dict:
                        if (
                            dist_to_node
                            > visited_markings_distance_dict[new_mark].dist_from_origin
                        ) or (
                            dist_to_node
                            == visited_markings_distance_dict[new_mark].dist_from_origin
                            and sol_vec[transition_idx] < 0.999
                        ):
                            continue

                    if new_mark not in visited_markings_distance_dict:
                        need_push_node = True
                        node = search_node(
                            min_dist_node, transition, new_mark, dist_to_node
                        )
                        visited_markings_distance_dict[new_mark] = node

                    else:
                        need_heapify = True
                        node = visited_markings_distance_dict[new_mark]
                        node.dist_from_origin = dist_to_node
                        node.ancestor = min_dist_node
                        node.transition_to_ancestor = transition

                    node.heuristic_distance = max(
                        0, min_dist_node.heuristic_distance - transition.weight
                    )
                    node.total_distance = (
                        node.heuristic_distance + node.dist_from_origin
                    )

                    if min_dist_node.solution_vec[transition_idx] >= 0.999:
                        new_sol_vec = np.array(min_dist_node.solution_vec, copy=True)

                        if min_dist_node.solution_vec[transition_idx] >= 1:
                            new_sol_vec[transition_idx] -= 1

                        else:
                            new_sol_vec[transition_idx] = 0

                        node.solution_vec = new_sol_vec
                        node.have_exact_known_solution = True
                        node.have_estimated_solution = False

                    else:
                        node.have_exact_known_solution = False
                        node.have_estimated_solution = True
                        node.solution_vec = None

                    node.disappointing = min_dist_node.disappointing

                    if transition.move_type in {"sync", "trace"}:
                        node.n_explained_events = min_dist_node.n_explained_events + 1

                    else:
                        node.n_explained_events = min_dist_node.n_explained_events

                    if need_push_node:
                        heappush(distance_min_heap, node)

            if need_heapify:
                heapify(distance_min_heap)

        curr_node = min_dist_node
        path = []

        while curr_node.ancestor:
            path.append(curr_node.transition_to_ancestor.name)
            curr_node = curr_node.ancestor

        print(
            f"Optimal alignment cost: {min_dist_node.dist_from_origin }"
        )  # , \n Optimal alignment: \n {path[::-1]}')
        return path[::-1], min_dist_node.dist_from_origin

    def _fire_transition(self, mark, transition):
        """Input: Mark object, Transition object
        Output: Marking object"""

        subtract_mark = [0] * len(mark.places)
        for arc in transition.in_arcs:
            place_idx = self.places_indices[arc.source.name]
            subtract_mark[place_idx] -= arc.weight

        add_mark = [0] * len(mark.places)
        for arc in transition.out_arcs:
            place_idx = self.places_indices[arc.target.name]
            add_mark[place_idx] += arc.weight

        new_mark = tuple(
            [sum(x) for x in zip(list(mark.places), subtract_mark, add_mark)]
        )
        for elem in new_mark:
            if elem < 0:
                print(
                    f"the mark was: {mark} and I subtract the following values: {subtract_mark} and adding these: {add_mark} \
                which results in this: {new_mark} and all this sh!t was by using this transition: {transition.name}"
                )
        new_mark_obj = Marking(new_mark)

        return new_mark_obj

    #     # original function
    #     def __update_sync_product_trans_names(self, sync_product=None):
    #         for trans in sync_product.transitions:
    #             if trans.move_type == 'model':
    #                 trans.name = f'({trans.name}, >>)'
    #             elif trans.move_type == 'trace':
    #                 trans.name = f'(>>, {trans.name})'
    #             else:
    #                 trans.name = f'({trans.name}, {trans.name})'

    #         sync_product.transitions_indices = {self.transitions[i].name:i for i in range(len(self.transitions))}

    #         return sync_product

    def _update_sync_product_trans_names(self, sync_product=None):
        if sync_product is None:
            transitions = self.transitions
        else:
            transitions = sync_product.transitions

        for trans in transitions:
            if trans.move_type == "model":
                trans.name = f"({trans.name}, >>)"
            elif trans.move_type == "trace":
                trans.name = f"(>>, {trans.name})"
            else:
                trans.name = f"({trans.name}, {trans.name})"

        transitions_indices = {transitions[i].name: i for i in range(len(transitions))}

        if sync_product is not None:
            sync_product.transitions_indices = transitions_indices
            return sync_product
        else:
            self.transitions_indices = transitions_indices

    def compute_conditioned_weight(self, path_prefix, transition, prob_dict, lamda=0.5):

        if prob_dict is None:
            return transition.weight

        print(f"prob dict = {prob_dict}")
        if transition.label is None:
            #             print(f'Transition label=None thus returning weight of 0')
            return 0

        #         print(f'original trans weight={transition.weight}')
        transition_weight = transition.weight
        transition_label = transition.label
        full_path = path_prefix + transition_label
        #         print(f'The full path including the transition label={full_path}')

        if path_prefix == "":
            return transition_weight

        if full_path in prob_dict:
            #             print(f'full path={full_path} is in the prob dict and conditioned weight= {0.5*(1-prob_dict[full_path]) + 0.5*transition_weight}')
            #             return (1-lamda)*(1-prob_dict[full_path]) + lamda*transition_weight
            return (1 - lamda) * (
                (1 - prob_dict[full_path]) * transition_weight
            ) + lamda * transition_weight

        #         print(f'full path={full_path} is not in the prob dict.. sorry.. ')
        longest_prefix = self.find_longest_prefix(full_path, prob_dict)

        if longest_prefix:
            #             print(f'longest prefix={longest_prefix} is in dict! The conditioned weight= {0.5*(1-prob_dict[longest_prefix]) + 0.5*transition_weight}')
            #             return (1-lamda)*(1-prob_dict[longest_prefix]) + lamda*transition_weight
            return (1 - lamda) * (
                (1 - prob_dict[longest_prefix]) * transition_weight
            ) + lamda * transition_weight
        #         print(f'no prefix exists for {full_path}..conditioned weight= {0.5 + 0.5*transition_weight}')
        #         return (1-lamda) + lamda*transition_weight
        return transition_weight

    def find_longest_prefix(self, full_path, prob_dict):
        longest_prefix = None
        for i in range(len(full_path) - 1):
            if full_path[i:] in prob_dict:
                return full_path[i:]
        #             print(f'prefix={full_path[i:]} is not in the dict')

        return longest_prefix
