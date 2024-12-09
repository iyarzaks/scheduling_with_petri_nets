import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import pandas as pd
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils


@dataclass
class TransitionMetadata:
    duration: int
    resources: Dict[str, int]


class AugmentedPetriNet:
    def __init__(self, net: PetriNet):
        self.net = net
        self.transition_metadata: Dict[PetriNet.Transition, TransitionMetadata] = {}
        self.resource_places: Dict[str, PetriNet.Place] = {}

    def add_transition_metadata(
        self, transition: PetriNet.Transition, duration: int, resources: Dict[str, int]
    ):
        self.transition_metadata[transition] = TransitionMetadata(duration, resources)

    def get_metadata(
        self, transition: PetriNet.Transition
    ) -> Optional[TransitionMetadata]:
        return self.transition_metadata.get(transition)

    def add_resource_places(self, available_resources: Dict[str, int]):
        """Add places for renewable resources and connect them to transitions."""
        # Create resource places
        for resource_type, amount in available_resources.items():
            place = PetriNet.Place(f"resource_{resource_type}")
            self.net.places.add(place)
            self.resource_places[resource_type] = place

        # Connect resources to transitions
        for transition in self.net.transitions:
            metadata = self.get_metadata(transition)
            if metadata:
                for resource_type, amount in metadata.resources.items():
                    resource_place = self.resource_places[resource_type]
                    # Add arc from resource place to transition
                    petri_utils.add_arc_from_to(
                        resource_place, transition, self.net, weight=amount
                    )
                    # Add arc from transition back to resource place
                    petri_utils.add_arc_from_to(
                        transition, resource_place, self.net, weight=amount
                    )


class State:
    def __init__(
        self,
        marking: Marking,
        time: int = 0,
        active_transitions: Dict[PetriNet.Transition, int] = None,
    ):
        self.marking = marking
        self.time = time
        # Keep track of currently running transitions and their completion times
        self.active_transitions = active_transitions or {}

    def __lt__(self, other):
        return self.time < other.time

    def __eq__(self, other):
        return (
            self.marking == other.marking
            and self.time == other.time
            and self.active_transitions == other.active_transitions
        )

    def __hash__(self):
        return hash(
            (
                frozenset(self.marking.items()),
                self.time,
                frozenset(self.active_transitions.items()),
            )
        )


def check_resource_availability(
    current_resources: Dict[str, int],
    required_resources: Dict[str, int],
    available_resources: Dict[str, int],
) -> bool:
    """Check if enough resources are available for a new activity."""
    for resource, amount in required_resources.items():
        if current_resources.get(resource, 0) + amount > available_resources.get(
            resource, 0
        ):
            return False
    return True


def calculate_resource_usage(
    active_transitions: Dict[PetriNet.Transition, int], aug_net: AugmentedPetriNet
) -> Dict[str, int]:
    """Calculate current resource usage from active transitions."""
    resource_usage = {}
    for transition in active_transitions:
        metadata = aug_net.get_metadata(transition)
        if metadata:
            for resource, amount in metadata.resources.items():
                resource_usage[resource] = resource_usage.get(resource, 0) + amount
    return resource_usage


def get_activity_metadata(csv_path: str) -> Dict[str, Dict]:
    """Read activity metadata from CSV file."""
    df = pd.read_csv(csv_path)

    activity_metadata = {}
    for _, row in df.iterrows():
        activity = str(row["concept:name"])
        if activity not in activity_metadata:
            activity_metadata[activity] = {
                "duration": int(str(row["Duration"]).replace("d", "")),
                "resources": {
                    "R1": int(row["resource_demand_R1"]),
                    "R2": int(row["resource_demand_R2"]),
                },
            }
    return activity_metadata


def augment_petri_net(net: PetriNet, metadata: Dict[str, Dict]) -> AugmentedPetriNet:
    """Create augmented Petri net from existing net and metadata."""
    aug_net = AugmentedPetriNet(net)

    for transition in net.transitions:
        if transition.label in metadata:
            meta = metadata[transition.label]
            aug_net.add_transition_metadata(
                transition, duration=meta["duration"], resources=meta["resources"]
            )

    return aug_net


def is_marking_greater_or_equal(
    marking1: Marking, goal_marking: Marking, required_places: Set[PetriNet.Place]
) -> bool:
    """Check if marking1 satisfies goal conditions only for required places."""
    return all(
        marking1.get(place, 0) >= goal_marking.get(place, 0)
        for place in required_places
    )


def get_enabled_transitions(
    net: PetriNet, marking: Marking
) -> List[PetriNet.Transition]:
    """Get all enabled transitions for current marking."""
    enabled = []
    for transition in net.transitions:
        if is_transition_enabled(transition, marking):
            enabled.append(transition)
    return enabled


def is_transition_enabled(transition: PetriNet.Transition, marking: Marking) -> bool:
    """Check if a transition is enabled."""
    for arc in transition.in_arcs:
        if arc.source not in marking or marking[arc.source] < arc.weight:
            return False
    return True


def fire_transition(transition: PetriNet.Transition, marking: Marking) -> Marking:
    """Fire a transition and return new marking."""
    new_marking = Marking(marking)

    # Remove tokens from input places
    for arc in transition.in_arcs:
        new_marking[arc.source] -= arc.weight
        if new_marking[arc.source] == 0:
            del new_marking[arc.source]

    # Add tokens to output places
    for arc in transition.out_arcs:
        if arc.target not in new_marking:
            new_marking[arc.target] = 0
        new_marking[arc.target] += arc.weight

    return new_marking


def create_augmented_petri_net(
    net: PetriNet, im, metadata: Dict[str, Dict], available_resources: Dict[str, int]
) -> Tuple[AugmentedPetriNet, Marking]:
    """Create augmented Petri net with resource places."""
    aug_net = AugmentedPetriNet(net)

    # Add metadata to transitions
    for transition in net.transitions:
        if transition.label in metadata:
            meta = metadata[transition.label]
            aug_net.add_transition_metadata(
                transition, duration=meta["duration"], resources=meta["resources"]
            )
        else:
            aug_net.add_transition_metadata(transition, duration=0, resources={})

    # Add resource places and arcs
    aug_net.add_resource_places(available_resources)

    # Create initial marking including resources
    initial_marking = Marking()
    for place, count in im.items():
        initial_marking[place] = count

    # Add tokens for resources
    for resource_type, amount in available_resources.items():
        resource_place = aug_net.resource_places[resource_type]
        initial_marking[resource_place] = amount

    return aug_net, initial_marking


def get_reachable_transitions(
    place: PetriNet.Place, net: PetriNet, visited_places: Set[PetriNet.Place] = None
) -> Set[PetriNet.Transition]:
    """
    Get all transitions that are reachable from a place in the Petri net.
    Uses DFS to find all possible transitions.
    """
    if visited_places is None:
        visited_places = set()

    if place in visited_places:
        return set()

    visited_places.add(place)
    reachable = set()

    # Get immediate transitions
    for transition in net.transitions:
        if any(arc.source == place for arc in transition.in_arcs):
            reachable.add(transition)
            # Add transitions reachable from output places
            for arc in transition.out_arcs:
                reachable.update(
                    get_reachable_transitions(arc.target, net, visited_places)
                )

    return reachable


def get_min_path_duration(
    aug_net: AugmentedPetriNet,
    from_place: PetriNet.Place,
    to_place: PetriNet.Place,
    visited: Set[PetriNet.Place] = None,
) -> int:
    """Calculate minimum duration from one place to another."""
    if visited is None:
        visited = set()

    if from_place in visited:
        return float("inf")

    if from_place == to_place:
        return 0

    visited.add(from_place)
    min_duration = float("inf")

    # Try all possible next transitions
    for trans in aug_net.net.transitions:
        if any(arc.source == from_place for arc in trans.in_arcs):
            metadata = aug_net.get_metadata(trans)
            trans_duration = metadata.duration if metadata else 0

            # Try all output places of this transition
            for arc in trans.out_arcs:
                next_duration = get_min_path_duration(
                    aug_net, arc.target, to_place, visited.copy()
                )
                if next_duration != float("inf"):
                    path_duration = trans_duration + next_duration
                    min_duration = min(min_duration, path_duration)

    return min_duration


def get_sequential_duration(
    aug_net: AugmentedPetriNet, join_trans: PetriNet.Transition
) -> int:
    """Calculate sequential path duration from join to end."""
    total_duration = 0
    current = join_trans

    # Start with the join transition
    metadata = aug_net.get_metadata(current)
    if metadata:
        total_duration += metadata.duration

    # Follow path to end
    while current:
        output_places = [arc.target for arc in current.out_arcs]
        if not output_places:
            break

        next_trans = None
        for place in output_places:
            # Skip resource places
            if place.name.startswith("resource_"):
                continue

            for trans in aug_net.net.transitions:
                if any(arc.source == place for arc in trans.in_arcs):
                    metadata = aug_net.get_metadata(trans)
                    if metadata:
                        total_duration += metadata.duration
                    next_trans = trans
                    break
            if next_trans:
                break

        if not next_trans:
            break
        current = next_trans

    return total_duration


from typing import Set


class CachedPetriNet:
    def __init__(self, aug_net, debug=False):
        self.aug_net = aug_net
        self.debug = debug

    def get_duration(self, transition) -> int:
        if transition.name.startswith("skip_"):
            if self.debug:
                print(f"Skip transition {transition.name}: duration = 0")
            return 0
        duration = (
            self.aug_net.get_metadata(transition).duration
            if self.aug_net.get_metadata(transition)
            else 0
        )
        if self.debug:
            print(f"Transition {transition.name}: duration = {duration}")
        return duration

    def get_path_duration(self, from_place: str, to_place: str) -> float:
        if self.debug:
            print(f"\nCalculating path from {from_place} to {to_place}")
        distances = {from_place: 0}
        visited = set()
        queue = [(0, from_place)]
        heapq.heapify(queue)

        while queue:
            current_dist, current_place = heapq.heappop(queue)
            if self.debug:
                print(f"Processing place {current_place} at distance {current_dist}")

            if current_place == to_place:
                if self.debug:
                    print(f"Found path to target! Distance = {current_dist}")
                return current_dist

            if current_place in visited:
                continue

            visited.add(current_place)

            for trans in self.aug_net.net.transitions:
                if not any(arc.source.name == current_place for arc in trans.in_arcs):
                    continue

                if self.debug:
                    print(f"  Found outgoing transition: {trans.name}")
                duration = self.get_duration(trans)

                for out_arc in trans.out_arcs:
                    out_place = out_arc.target.name
                    if out_place.startswith("resource_"):
                        if self.debug:
                            print(f"    Skipping resource place {out_place}")
                        continue

                    new_dist = current_dist + duration
                    if self.debug:
                        print(f"    To place {out_place}, new distance = {new_dist}")

                    if out_place not in distances or new_dist < distances[out_place]:
                        distances[out_place] = new_dist
                        heapq.heappush(queue, (new_dist, out_place))

        if self.debug:
            print(f"No path found from {from_place} to {to_place}")
        return float("inf")


# def optimized_heuristic(
#     current_state: State, aug_net: AugmentedPetriNet, debug: bool = False
# ) -> int:
#     """Calculate minimum completion time assuming unlimited resources."""
#
#     def get_independent_paths(place, visited=None, path=None):
#         """Get all independent paths from a place to sink."""
#         if visited is None:
#             visited = set()
#         if path is None:
#             path = []
#
#         if place.name == "sink":
#             return [path]
#
#         paths = []
#         for arc in place.out_arcs:
#             transition = arc.target
#             if transition not in visited:
#                 new_visited = visited | {transition}
#                 new_path = path + [transition]
#
#                 for out_arc in transition.out_arcs:
#                     if not out_arc.target.name.startswith("resource_"):
#                         paths.extend(
#                             get_independent_paths(out_arc.target, new_visited, new_path)
#                         )
#
#         return paths
#
#     def path_duration(path):
#         """Calculate duration of a path."""
#         return sum(
#             aug_net.get_metadata(t).duration if aug_net.get_metadata(t) else 0
#             for t in path
#         )
#
#     def are_parallel(path1, path2):
#         """Check if two paths can run in parallel (no dependencies)."""
#         return not (set(path1) & set(path2))
#
#     # Get active transition completion time
#     active_completion_time = (
#         max(current_state.active_transitions.values()) - current_state.time
#         if current_state.active_transitions
#         else 0
#     )
#
#     # Get all current places with tokens
#     current_places = [
#         p
#         for p, tokens in current_state.marking.items()
#         if tokens > 0 and not p.name.startswith("resource_")
#     ]
#
#     # Get all possible paths from current marking to sink
#     all_paths = []
#     for place in current_places:
#         all_paths.extend(get_independent_paths(place))
#
#     # Group parallel paths
#     parallel_groups = []
#     for path in all_paths:
#         # Find or create a group for this path
#         added = False
#         for group in parallel_groups:
#             if all(are_parallel(path, p) for p in group):
#                 group.append(path)
#                 added = True
#                 break
#         if not added:
#             parallel_groups.append([path])
#
#     # Calculate duration for each group (max of parallel paths)
#     group_durations = []
#     for group in parallel_groups:
#         group_duration = max(path_duration(path) for path in group)
#         group_durations.append(group_duration)
#
#     # Total duration is max of parallel groups
#     total_duration = max(group_durations) if group_durations else 0
#
#     return max(active_completion_time, total_duration)


def segment_heuristic(
    current_state: State, aug_net: AugmentedPetriNet, target_place: str
) -> int:
    """
    Calculate heuristic value for A* search to reach the next split point or final marking.

    Args:
        current_state: Current state of the search
        aug_net: Augmented Petri net
        target_place: Name of the target place (split point) we're trying to reach
    """
    # Handle active transitions
    active_completion_time = (
        max(current_state.active_transitions.values()) - current_state.time
        if current_state.active_transitions
        else 0
    )

    # Find paths to target place
    parallel_durations = []
    target = next(p for p in aug_net.net.places if p.name == target_place)

    # For each marked place, calculate minimum path to target
    for marked_place, tokens in current_state.marking.items():
        if tokens > 0 and not marked_place.name.startswith("resource_"):
            duration = get_min_path_duration(aug_net, marked_place, target)
            if duration != float("inf"):
                parallel_durations.append(duration)

    # Maximum of parallel paths
    path_duration = max(parallel_durations) if parallel_durations else 0

    return max(active_completion_time, path_duration)


def preprocess_obvious_skips(aug_net: AugmentedPetriNet) -> Set[PetriNet.Transition]:
    """
    Identifies transitions that should be ignored in A* search because they have
    parallel skip transitions with zero duration.

    Returns:
        Set of transitions that should be ignored in favor of their skip paths
    """
    transitions_to_ignore = set()

    # Find all skip transitions
    skip_transitions = {
        t for t in aug_net.net.transitions if t.name and t.name.startswith("skip_")
    }

    for skip in skip_transitions:
        # Get input and output places for skip transition
        skip_inputs = {arc.source for arc in skip.in_arcs}
        skip_outputs = {arc.target for arc in skip.out_arcs}

        # Find parallel transitions by checking transitions that share input/output places
        for transition in aug_net.net.transitions:
            if (
                transition == skip
                or transition.name
                and transition.name.startswith("skip_")
            ):
                continue

            # Get input and output places for current transition
            trans_inputs = {arc.source for arc in transition.in_arcs}
            trans_outputs = {arc.target for arc in transition.out_arcs}

            # Check if transitions share same input and output places
            if trans_inputs.intersection(skip_inputs) and trans_outputs.intersection(
                skip_outputs
            ):
                metadata = aug_net.get_metadata(transition)
                if metadata and metadata.duration > 0:
                    transitions_to_ignore.add(transition)
                    # print(
                    #     f"Debug: Found parallel transition {transition.label} to skip {skip.name}"
                    # )
                    # print(f"  Skip inputs: {[p.name for p in skip_inputs]}")
                    # print(f"  Trans inputs: {[p.name for p in trans_inputs]}")
                    # print(f"  Skip outputs: {[p.name for p in skip_outputs]}")
                    # print(f"  Trans outputs: {[p.name for p in trans_outputs]}")

    return transitions_to_ignore


from dataclasses import dataclass
from typing import Dict, Set, List, Tuple
import heapq


@dataclass
class State:
    marking: Dict[str, int]
    time: float
    active_transitions: Dict["Transition", float]

    def get_available_resources(self, available_resources, aug_net):
        """Calculate currently available resources."""
        used = {}
        for t in self.active_transitions:
            metadata = aug_net.get_metadata(t)
            if metadata and metadata.resources:
                for resource, amount in metadata.resources.items():
                    used[resource] = used.get(resource, 0) + amount

        return {
            resource: total - used.get(resource, 0)
            for resource, total in available_resources.items()
        }

    def get_state_key(self, available_resources, aug_net):
        """Generate state key including marking, active transitions, and resource availability."""
        resources = self.get_available_resources(available_resources, aug_net)
        return (
            frozenset(self.marking.items()),
            frozenset(self.active_transitions.items()),
            frozenset(resources.items()),
            self.time,  # Include time in key
        )


class SearchNode:
    def __init__(self, f_score, g_score, state, path, completed):
        self.f_score = f_score
        self.g_score = g_score
        self.state = state
        self.path = path
        self.completed = completed

    def __lt__(self, other):
        if abs(self.f_score - other.f_score) > 1e-10:
            return self.f_score < other.f_score
        if abs(self.g_score - other.g_score) > 1e-10:
            return self.g_score > other.g_score
        return self.state.time > other.state.time


def get_enabled_transitions_with_resources(
    aug_net, state, available_resources, completed
):
    """Get transitions that are enabled by marking and have resources available."""
    enabled = get_enabled_transitions(aug_net.net, state.marking)
    if not isinstance(enabled, (set, list)):
        enabled = {enabled}

    # Calculate resources currently in use
    used_resources = defaultdict(int)
    for t in state.active_transitions:
        metadata = aug_net.get_metadata(t)
        if metadata and metadata.resources:
            for resource, amount in metadata.resources.items():
                used_resources[resource] += amount

    # Calculate remaining resources
    remaining_resources = {
        resource: total - used_resources.get(resource, 0)
        for resource, total in available_resources.items()
    }

    # Filter transitions
    return {
        t
        for t in enabled
        if t not in completed
        and t not in state.active_transitions
        and all(
            amount <= remaining_resources.get(resource, 0)
            for resource, amount in (aug_net.get_metadata(t).resources or {}).items()
        )
    }


def optimized_heuristic(current_state: State, aug_net: AugmentedPetriNet) -> float:
    """Calculate minimum completion time assuming unlimited resources."""

    def get_independent_paths(place, visited=None, path=None, start_time=0):
        """Get all independent paths from a place to sink."""
        if visited is None:
            visited = set()
        if path is None:
            path = []

        if place.name == "sink":
            return [(path, start_time)]

        paths = []
        for arc in place.out_arcs:
            transition = arc.target
            if transition not in visited:
                new_visited = visited | {transition}

                # If transition is active, use remaining duration
                if transition in current_state.active_transitions:
                    remaining_time = (
                        current_state.active_transitions[transition]
                        - current_state.time
                    )
                    new_path = path + [(transition, remaining_time)]
                else:
                    duration = (
                        aug_net.get_metadata(transition).duration
                        if aug_net.get_metadata(transition)
                        else 0
                    )
                    new_path = path + [(transition, duration)]

                for out_arc in transition.out_arcs:
                    if not out_arc.target.name.startswith("resource_"):
                        paths.extend(
                            get_independent_paths(
                                out_arc.target, new_visited, new_path, start_time
                            )
                        )

        return paths

    def path_duration(path):
        """Calculate duration of a path."""
        return sum(duration for _, duration in path)

    def are_parallel(path1, path2):
        """Check if two paths can run in parallel (no dependencies)."""
        transitions1 = set(t for t, _ in path1)
        transitions2 = set(t for t, _ in path2)
        return not (transitions1 & transitions2)

    # Get all current places with tokens
    current_places = [
        p
        for p, tokens in current_state.marking.items()
        if tokens > 0 and not p.name.startswith("resource_")
    ]

    # Get all possible paths from current marking to sink
    all_paths = []
    for place in current_places:
        all_paths.extend(get_independent_paths(place))

    # Group parallel paths
    parallel_groups = []
    for path, start_time in all_paths:
        added = False
        for group in parallel_groups:
            if all(are_parallel(path, p) for p, _ in group):
                group.append((path, start_time))
                added = True
                break
        if not added:
            parallel_groups.append([(path, start_time)])

    # Calculate duration for each group (max of parallel paths)
    group_durations = []
    for group in parallel_groups:
        group_duration = max(path_duration(path) for path, _ in group)
        group_durations.append(group_duration)

    # Total duration is max of parallel groups
    total_duration = max(group_durations) if group_durations else 0

    return total_duration


def astar_search(
    aug_net, initial_marking, goal_marking, available_resources, debug=False
):
    """A* search with proper handling of resources and parallel execution."""

    def get_enabled_transitions_with_resources(
        aug_net, state, available_resources, completed
    ):
        """Get transitions that are enabled by marking and have resources available."""
        enabled = get_enabled_transitions(aug_net.net, state.marking)
        if not isinstance(enabled, (set, list)):
            enabled = {enabled}

        # Calculate resources currently in use
        used_resources = defaultdict(int)
        for t in state.active_transitions:
            metadata = aug_net.get_metadata(t)
            if metadata and metadata.resources:
                for resource, amount in metadata.resources.items():
                    used_resources[resource] += amount

        # Calculate remaining resources
        remaining_resources = {
            resource: total - used_resources.get(resource, 0)
            for resource, total in available_resources.items()
        }

        # Filter transitions
        return {
            t
            for t in enabled
            if t not in completed
            and t not in state.active_transitions
            and all(
                amount <= remaining_resources.get(resource, 0)
                for resource, amount in (
                    aug_net.get_metadata(t).resources or {}
                ).items()
            )
        }

    def try_start_transitions(state, transitions, current_time, path, completed):
        """Try all valid combinations of starting transitions."""
        nodes = []

        # Calculate resources currently in use by active transitions
        active_resources = defaultdict(int)
        for t in state.active_transitions:
            metadata = aug_net.get_metadata(t)
            if metadata and metadata.resources:
                for resource, amount in metadata.resources.items():
                    active_resources[resource] += amount

        # Calculate remaining available resources
        remaining_resources = {
            resource: total - active_resources.get(resource, 0)
            for resource, total in available_resources.items()
        }

        # Try combinations
        for size in range(1, len(transitions) + 1):
            for combo in itertools.combinations(transitions, size):
                # Track total resources needed for this combination
                combo_resources = defaultdict(int)
                can_run = True

                # Sum up resources needed for all transitions in combo
                for transition in combo:
                    metadata = aug_net.get_metadata(transition)
                    if metadata and metadata.resources:
                        for resource, amount in metadata.resources.items():
                            combo_resources[resource] += amount
                            # Check if we'd exceed available resources
                            if combo_resources[resource] > remaining_resources.get(
                                resource, 0
                            ):
                                can_run = False
                                break
                    if not can_run:
                        break

                if can_run:
                    # Create new state with all transitions in combination running
                    new_active = dict(state.active_transitions)
                    for transition in combo:
                        metadata = aug_net.get_metadata(transition)
                        completion_time = current_time + (
                            metadata.duration if metadata else 0
                        )
                        new_active[transition] = completion_time

                    new_state = State(state.marking.copy(), current_time, new_active)
                    g_score = current_time
                    h_score = optimized_heuristic(new_state, aug_net)
                    f_score = g_score + h_score

                    nodes.append(
                        SearchNode(
                            f_score, g_score, new_state, path.copy(), completed.copy()
                        )
                    )

        return nodes

    initial_state = State(initial_marking, 0, {})
    initial_node = SearchNode(0, 0, initial_state, [], set())

    open_set = [initial_node]
    heapq.heapify(open_set)
    closed_set = set()

    while open_set:
        current_node = heapq.heappop(open_set)
        current_state = current_node.state

        if debug:
            print(f"\n=== Exploring state at time {current_state.time} ===")
            print(
                f"Active: {[(t.name, time) for t, time in current_state.active_transitions.items()]}"
            )
            print(f"Completed: {[t.name for t in current_node.completed]}")

        # Check if goal reached
        if (
            all(
                current_state.marking.get(p, 0) >= goal_marking.get(p, 0)
                for p in goal_marking
            )
            and not current_state.active_transitions
        ):
            print("\nGoal reached!")
            return current_node.path

        state_key = current_state.get_state_key(available_resources, aug_net)
        if state_key in closed_set:
            continue

        closed_set.add(state_key)

        # Complete transitions that finish next
        if current_state.active_transitions:
            next_completion_time = min(current_state.active_transitions.values())
            completing = {
                t: time
                for t, time in current_state.active_transitions.items()
                if abs(time - next_completion_time) < 1e-10
            }

            if debug:
                print(
                    f"Completing at {next_completion_time}: {[t.name for t in completing]}"
                )

            # Create new state after completions
            new_marking = current_state.marking.copy()
            for transition in completing:
                new_marking = fire_transition(transition, new_marking)

            remaining_active = {
                t: time
                for t, time in current_state.active_transitions.items()
                if t not in completing
            }

            # Get newly enabled transitions
            intermediate_state = State(
                new_marking, next_completion_time, remaining_active
            )
            enabled = get_enabled_transitions_with_resources(
                aug_net,
                intermediate_state,
                available_resources,
                current_node.completed.union(completing.keys()),
            )

            if enabled:
                # Try starting new transitions at completion time
                new_completed = current_node.completed.union(completing.keys())
                new_path = current_node.path + [
                    (
                        t,
                        (
                            time - aug_net.get_metadata(t).duration
                            if aug_net.get_metadata(t)
                            else 0
                        ),
                    )
                    for t, time in completing.items()
                ]

                new_nodes = try_start_transitions(
                    intermediate_state,
                    enabled,
                    next_completion_time,
                    new_path,
                    new_completed,
                )

                for node in new_nodes:
                    heapq.heappush(open_set, node)

            # Also consider just completing without starting new transitions
            g_score = next_completion_time
            h_score = optimized_heuristic(intermediate_state, aug_net)
            f_score = g_score + h_score

            complete_node = SearchNode(
                f_score,
                g_score,
                intermediate_state,
                current_node.path
                + [
                    (
                        t,
                        (
                            time - aug_net.get_metadata(t).duration
                            if aug_net.get_metadata(t)
                            else 0
                        ),
                    )
                    for t, time in completing.items()
                ],
                current_node.completed.union(completing.keys()),
            )
            heapq.heappush(open_set, complete_node)

        # Try starting new transitions at current time if none active
        elif not current_state.active_transitions:
            enabled = get_enabled_transitions_with_resources(
                aug_net, current_state, available_resources, current_node.completed
            )

            if enabled:
                new_nodes = try_start_transitions(
                    current_state,
                    enabled,
                    current_state.time,
                    current_node.path,
                    current_node.completed,
                )

                for node in new_nodes:
                    heapq.heappush(open_set, node)

    print("\nNo solution found!")
    return None


def analyze_final_marking(net: PetriNet, fm: Marking):
    """Analyze which places are actually required in final marking."""
    required_places = set()
    for place, count in fm.items():
        if count > 0:
            required_places.add(place)
    return required_places


def analyze_path(
    path: List[Tuple[PetriNet.Transition, int]],
    aug_net: AugmentedPetriNet,
    total_transitions: Set[PetriNet.Transition],
):
    """Enhanced path analysis showing parallel activities."""
    if not path:
        print("No valid path found.")
        return

    # Sort activities by start time
    sorted_path = sorted(path, key=lambda x: x[1])

    print("\nPath Analysis:")
    print("-" * 90)
    print(
        f"{'Activity':<20} {'Start Time':<12} {'Duration':<10} {'End Time':<10} {'Resources':<15} {'Parallel With'}"
    )
    print("-" * 90)

    activity_times = {}  # Store start and end times for each activity
    label_to_transition = {}  # Map labels to transitions

    # First pass: collect all timing information
    for transition, start_time in sorted_path:
        metadata = aug_net.get_metadata(transition)
        if metadata and transition.label:  # Only store labeled transitions
            end_time = start_time + metadata.duration
            activity_times[transition] = (start_time, end_time)
            if transition.label:
                label_to_transition[transition.label] = transition

    # Second pass: print with parallel activity information
    total_duration = 0
    resource_usage = {}

    for transition, start_time in sorted_path:
        metadata = aug_net.get_metadata(transition)
        if (
            transition.label and metadata
        ):  # Skip unlabeled transitions and those without metadata
            end_time = start_time + metadata.duration
            total_duration = max(total_duration, end_time)

            # Find parallel activities
            parallel_activities = []
            for other_trans, (other_start, other_end) in activity_times.items():
                if (
                    other_trans != transition and other_trans.label
                ):  # Ensure other transition has label
                    if other_start < end_time and start_time < other_end:
                        if other_trans.label:  # Double check label exists
                            parallel_activities.append(other_trans.label)

            # Handle resources string
            resources_str = ", ".join(f"{r}:{a}" for r, a in metadata.resources.items())

            # Filter out None values and create parallel string
            parallel_activities = [
                act for act in parallel_activities if act is not None
            ]
            parallel_str = (
                ", ".join(parallel_activities) if parallel_activities else "None"
            )

            print(
                f"{transition.label:<20} {start_time:<12} {metadata.duration:<10} "
                f"{end_time:<10} {resources_str:<15} {parallel_str}"
            )

            for resource, amount in metadata.resources.items():
                resource_usage[resource] = resource_usage.get(resource, 0) + amount

    print("-" * 90)
    print(f"Total duration: {total_duration}")
    print("\nResource usage:")
    for resource, amount in resource_usage.items():
        print(f"- {resource}: {amount}")

    return total_duration


def find_split_points(net, num_splits: int = None) -> List[str]:
    """
    Find true bottleneck places in the PM4Py Petri net where all paths must converge.

    Args:
        aug_net: The augmented Petri net
        num_splits: Optional maximum number of split points to return

    Returns:
        List of place IDs that are bottlenecks
    """
    # Build connection maps for easier traversal
    place_to_transitions = defaultdict(list)  # place -> list of output transitions
    transition_to_places = defaultdict(list)  # transition -> list of output places

    # Build the maps using arc information
    for transition in net.transitions:
        # Get input places from in_arcs
        for arc in transition.in_arcs:
            place = arc.source
            place_to_transitions[place].append(transition)

        # Get output places from out_arcs
        for arc in transition.out_arcs:
            place = arc.target
            transition_to_places[transition].append(place)

    def get_all_places():
        """Get all unique places in the net."""
        places = set()
        for transition in net.transitions:
            for arc in transition.in_arcs:
                places.add(arc.source)
            for arc in transition.out_arcs:
                places.add(arc.target)
        return places

    def get_initial_places():
        """Find places with no incoming transitions."""
        all_places = get_all_places()
        has_inputs = set()
        for transition in net.transitions:
            for arc in transition.out_arcs:
                has_inputs.add(arc.target)
        return all_places - has_inputs

    def get_final_places():
        """Find places with no outgoing transitions."""
        all_places = get_all_places()
        has_outputs = set()
        for transition in net.transitions:
            for arc in transition.in_arcs:
                has_outputs.add(arc.source)
        return all_places - has_outputs

    def get_reachable_places(start_places: Set, excluding_place=None) -> Set:
        """Get all places reachable from given places without going through excluding_place."""
        reachable = set()
        queue = list(start_places)

        while queue:
            place = queue.pop(0)
            if place in reachable or place == excluding_place:
                continue

            reachable.add(place)

            # Follow path forward through transitions
            for transition in net.transitions:
                # Check if this transition takes from our current place
                if any(arc.source == place for arc in transition.in_arcs):
                    # Add all output places of this transition
                    for arc in transition.out_arcs:
                        out_place = arc.target
                        if out_place not in reachable and out_place != excluding_place:
                            queue.append(out_place)

        return reachable

    def is_bottleneck(place) -> bool:
        """
        Check if a place is a true bottleneck by seeing if all paths from
        initial to final places must go through it.
        """
        initial_places = get_initial_places()
        final_places = get_final_places()

        # Skip if it's an initial or final place
        if place in initial_places or place in final_places:
            return False

        # Get reachable places without going through this place
        reachable = get_reachable_places(initial_places, excluding_place=place)

        # If no final place is reachable without this place, it's a bottleneck
        return not any(p in reachable for p in final_places)

    def get_path_count(place) -> int:
        """Count number of distinct paths going through this place."""
        incoming = len(
            [
                t
                for t in net.transitions
                if any(arc.target == place for arc in t.out_arcs)
            ]
        )
        outgoing = len(
            [
                t
                for t in net.transitions
                if any(arc.source == place for arc in t.in_arcs)
            ]
        )
        return incoming * outgoing

    # Find all bottlenecks
    bottlenecks = []
    for place in get_all_places():
        if is_bottleneck(place):
            bottlenecks.append((place.name, get_path_count(place)))

    # Sort bottlenecks by path count (most constrained first)
    bottlenecks.sort(key=lambda x: x[1], reverse=True)

    # Return only the place names, limited by num_splits if specified
    if num_splits is not None and num_splits < len(bottlenecks):
        return select_diverse_bottlenecks(bottlenecks, num_splits)
    return [b[0] for b in bottlenecks]


def select_diverse_bottlenecks(
    bottlenecks: List[Tuple[str, int]], num_splits: int
) -> List[str]:
    """
    Select diverse bottleneck places based on their numbered names (p_1, p_2, etc.).
    Tries to spread selections evenly across the net's structure.

    Args:
        bottlenecks: List of (place_name, path_count) tuples
        num_splits: Number of bottlenecks to select

    Returns:
        List of selected place names
    """
    if num_splits >= len(bottlenecks):
        return [b[0] for b in bottlenecks]

    # Extract place numbers from names (p_1 -> 1, p_2 -> 2, etc.)
    numbered_bottlenecks = []
    for place_name, path_count in bottlenecks:
        num = int(place_name.split("_")[1])
        numbered_bottlenecks.append((place_name, num, path_count))

    # Sort by number to get full range
    numbered_bottlenecks.sort(key=lambda x: x[1])
    total_places = numbered_bottlenecks[-1][1]

    # Calculate ideal spacing between selected bottlenecks
    ideal_gap = total_places / (num_splits + 1)

    # Select bottlenecks closest to ideal positions
    selected = []
    for i in range(num_splits):
        target_position = ideal_gap * (i + 1)
        # Find bottleneck closest to this position
        closest = min(numbered_bottlenecks, key=lambda x: abs(x[1] - target_position))
        selected.append(closest[0])
        numbered_bottlenecks.remove(closest)

    return selected
