# api_exploration_data.py

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property, lru_cache
import random
from typing import Any

from bidict import bidict
from loguru import logger
from rustworkx import PyDiGraph
from shapely.geometry import Point, MultiPoint

from api_common import memory


# Constants
GREAT_OCEAN_TERRITORY = 5
OQUILLAS_EYE_KEY = 1727
SUPER_ROOT = 99999
TILE_SCALE = 12800


class NodeType(IntEnum):
    """Exploration node types."""

    normal = 0
    village = 1
    city = 2
    gate = 3
    farm = 4
    trade = 5
    collect = 6
    quarry = 7
    logging = 8
    dangerous = 9
    finance = 10
    fish_trap = 11
    minor_finance = 12
    monopoly_farm = 13
    craft = 14
    excavation = 15
    count = 16


class GraphCoords:
    """
    Handles coordinate extraction and mapping between geographic (lat/lon)
    and scaled Cartesian (x/y) coordinates derived from graph node attributes x,z.

    NOTE: coord entries are based on nodes in super_graph.
    """

    def __init__(self, G, scale: float = TILE_SCALE):
        self.G = G
        self.scale = scale
        self.xz = {
            n: (G[n]["position"]["x"] / scale, G[n]["position"]["z"] / scale) for n in G.node_indices()
        }

    # Single-node accessors
    def as_cartesian(self, idx: int) -> tuple[float, float]:
        return self.xz[idx]

    def as_geographic(self, idx: int) -> tuple[float, float]:
        x, z = self.xz[idx]
        return (z, x)

    def as_cartesian_point(self, idx: int) -> Point:
        x, z = self.xz[idx]
        return Point(x, z)

    def as_geographic_point(self, idx: int) -> Point:
        lat, lon = self.as_geographic(idx)
        return Point(lon, lat)

    # Iterable versions
    def _indices(self, indices: Iterable[int] | None) -> Iterable[int]:
        return indices if indices is not None else self.xz.keys()

    def as_cartesians(self, indices: Iterable[int] | None = None) -> list[tuple[float, float]]:
        return [self.as_cartesian(i) for i in self._indices(indices)]

    def as_geographics(self, indices: Iterable[int] | None = None) -> list[tuple[float, float]]:
        return [self.as_geographic(i) for i in self._indices(indices)]

    def as_cartesian_points(self, indices: Iterable[int] | None = None) -> list[Point]:
        return [self.as_cartesian_point(i) for i in self._indices(indices)]

    def as_geographic_points(self, indices: Iterable[int] | None = None) -> list[Point]:
        return [self.as_geographic_point(i) for i in self._indices(indices)]

    def as_cartesian_multipoint(self, indices: Iterable[int] | None = None) -> MultiPoint:
        return MultiPoint(self.as_cartesian_points(indices))

    def as_geographic_multipoint(self, indices: Iterable[int] | None = None) -> MultiPoint:
        return MultiPoint(self.as_geographic_points(indices))


@dataclass
class ExplorationData:
    """Exploration data container with cached properties for the most common operations.

    Note: Methods are not cached.
    """

    data: dict[int, dict]
    hash: str

    @cached_property
    def graph(self) -> PyDiGraph:
        """Lazily generates the graph, leveraging the joblib function cache."""
        return _get_exploration_graph(self.data)

    @cached_property
    def super_graph(self) -> PyDiGraph:
        """Lazily generates the graph with injected SuperRoot, leveraging the joblib function cache."""
        from copy import deepcopy

        G = deepcopy(self.graph.copy())  # rustworkx copy() is shallow
        _inject_super_root(G, self.super_root)
        return G

    @cached_property
    def path_lengths(self) -> dict[tuple[int, int], int]:
        """Lazily calculates path lengths, leveraging the joblib function cache."""
        return _get_all_pairs_path_lengths(self.graph)

    @cached_property
    def cartesian_distances(self) -> dict[tuple[int, int], float]:
        """Lazily calculates cartesian distances, leveraging the joblib function cache."""
        return _get_all_pairs_cartesian_distances(self.data)

    @cached_property
    def coords(self) -> GraphCoords:
        """Scaled coordinate accessors."""
        return GraphCoords(self.super_graph)

    @cached_property
    def plantzones(self) -> list[int]:
        return [k for k, v in self.data.items() if v.get("is_workerman_plantzone")]

    @cached_property
    def max_plantzone_count(self) -> int:
        return len(self.plantzones)

    @cached_property
    def dangers(self) -> list[int]:
        return [k for k, v in self.data.items() if v.get("node_type") == NodeType.dangerous]

    @cached_property
    def max_danger_count(self) -> int:
        return len(self.dangers)

    @cached_property
    def towns(self) -> list[int]:
        return [n for n, d in self.data.items() if d.get("is_base_town", False)]

    @cached_property
    def capitals(self) -> dict[int, int]:
        return {
            self.data[t]["territory_key"]: t for t in self.towns if self.data[t]["node_type"] == NodeType.city
        }

    @cached_property
    def territories(self) -> set[int]:
        return set(n["territory_key"] for n in self.data.values())

    @cached_property
    def territory_towns(self) -> dict[int, list[int]]:
        tmp = {territory: [] for territory in self.territories}
        for t in self.towns:
            tmp[self.data[t]["territory_key"]].append(t)
        return tmp

    @cached_property
    def super_root(self) -> dict[str, Any]:
        """Returns a base town exploration node with waypoint_key: 99999 and
        link_list consisting of all other base_town nodes.

        This facilitates the connection of terminals through any potential root in the graph.

        NOTE: Adding super root to the graph breaks graph planarity!
        """
        link_list = self.towns
        return {
            "waypoint_key": 99999,
            "region_key": 99999,
            "region_group_key": 99999,
            "territory_key": 99999,
            "character_key": 99999,
            "node_type": 1,
            "is_town": True,
            "is_base_town": True,
            "is_plantzone": False,
            "is_workerman_plantzone": False,
            "is_warehouse_town": False,
            "is_worker_npc_town": False,
            "need_exploration_point": 0,
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "link_list": link_list,
            "worker_types": [],
            "region_houseinfo": {
                "has_rentable_lodging": False,
                "has_rentable_storage": False,
                "has_cashproduct_lodging": False,
                "has_cashproduct_storage": False,
            },
        }

    def path_length(self, u: int, v: int) -> int:
        return self.path_lengths[(u, v)]

    def cartesian_distance(self, u: int, v: int) -> float:
        return self.cartesian_distances[(u, v)]

    def towns_in_territory(self, territory: int) -> list[int]:
        return self.territory_towns[territory]

    def select_terminals(self, worker_percent: int, rng: random.Random) -> list[int]:
        num_workers = int(self.max_plantzone_count * worker_percent / 100)
        return rng.sample(self.plantzones, min(num_workers, self.max_plantzone_count))

    def select_dangers(self, selected_terminal_count: int, rng: random.Random) -> list[int]:
        # NOTE: The artificial limit of terminal count / 25 is to prevent to many dangers.
        danger_count = max(round(selected_terminal_count / 25), 1)
        return rng.sample(self.dangers, danger_count)


# This is a singleton, upon initialization it will invalidate the joblib cache
# if the exploration data has changed and then mem cache the result.
@lru_cache(maxsize=1)
def get_exploration_data():
    """Returns the ExplorationData instance for the current hash."""
    logger.trace("get_exploration_data")

    data, hash = _get_source_data_and_hash()
    logger.trace(f"  hash: {hash}")
    return _get_exploration_data(data, hash)


def _get_source_data_and_hash() -> tuple[dict[int, dict], str]:
    """
    Reads the raw exploration.json, computes its hash, and
    clears the dependent 'clean_exploration.json' file if the hash is new.

    Returns:
        tuple[dict, str]: The cleaned exploration data and its hash.
    """
    logger.trace("get_source_data_and_hash")
    import hashlib
    import json
    import api_data_store as ds

    source_filename = "exploration.json"

    content = ds.read_json(source_filename)
    encoded_content = json.dumps(content).encode()
    new_hash = hashlib.sha256(encoded_content).hexdigest()

    hash_file = f"{source_filename}.sha256"
    old_hash = ds.read_text(hash_file) if ds.is_file(hash_file) else None

    have_clean = ds.is_file("clean_exploration.json")

    if new_hash != old_hash and have_clean:
        ds.remove_file("clean_exploration.json")
    if new_hash != old_hash:
        ds.write_text(hash_file, new_hash)

    # Generate the cleaned data
    clean_data = _get_clean_exploration_data(new_hash)

    return clean_data, new_hash


@memory.cache
def _get_exploration_data(exploration_data: dict, new_hash: str):
    logger.trace("_get_exploration_data")
    return ExplorationData(exploration_data, new_hash)


@memory.cache
def _get_clean_exploration_data(hash_key: str):
    """Read exploration.json from data store and recursively remove
    - non valid entries from link lists
    - all nodes with an empty link list.

    Args:
        hash_key (str): Sole intent is cache invalidation

    NOTE: If 'clean_exploration.json' is present then it is used instead of being generated.
    """
    logger.trace("_get_clean_exploration_data")
    import api_data_store as ds

    filename = "clean_exploration.json"
    clean_data = ds.read_json(filename) if ds.is_file(filename) else None

    if clean_data:
        return clean_data

    data = ds.read_json("exploration.json")

    # NOTE: Great ocean nodes can not be traversed by workers.
    def non_base_great_ocean_node(k: int, v: dict) -> bool:
        return v["territory_key"] == GREAT_OCEAN_TERRITORY and not v["is_base_town"]

    data = {k: v for k, v in data.items() if not non_base_great_ocean_node(k, v)}

    # Recursive removals: previously removed nodes will have left entries in link lists
    while True:
        valid_keys = {k for k, v in data.items() if v["link_list"]}

        for v in data.values():
            v["link_list"] = [neighbor for neighbor in v["link_list"] if neighbor in valid_keys]

        new_data = {k: v for k, v in data.items() if v["link_list"]}
        if len(new_data) == len(data):
            break
        data = new_data

    ds.write_json(filename, data)

    return data


@memory.cache
def _get_exploration_graph(data: dict[int, dict]) -> PyDiGraph:
    """Generate and return a node weighted PyDiGraph graph from 'exploration.json'.

    - Graph will have anti-parallel bi-direcitonal edges and no self-loops.
    - Graph edges will have an edge weight equal to the head node's cost.

    NOTE: The returned graph will have an attribute [node_key_by_index] containing all
    node indices and exploration keys.
    """
    logger.debug("Generating node weighted directed graph...")

    graph = PyDiGraph(multigraph=False)

    # Add nodes to graph while mapping from node index to exploration key
    node_key_by_index = bidict({i: k for k, i in zip(data.keys(), graph.add_nodes_from(data.values()))})

    # Each node in exploration data has full link list info
    for i in graph.node_indices():
        neighbors = graph[i]["link_list"]
        for j in (node_key_by_index.inv[n] for n in neighbors):
            graph.add_edge(i, j, {"weight": graph[j]["need_exploration_point"]})

    graph.attrs = {"node_key_by_index": node_key_by_index}

    logger.debug(f"  generated graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges.")
    return graph


@memory.cache
def _get_all_pairs_shortest_paths(graph: PyDiGraph) -> dict[tuple[int, int], list[int]]:
    """Calculates and returns the node weighted shortest paths for all pairs.
    NOTE: output is graph indices not waypoint keys!
    """
    logger.trace("get_all_pairs_shortest_paths")

    from rustworkx import all_pairs_dijkstra_shortest_paths

    shortest_paths = all_pairs_dijkstra_shortest_paths(graph, edge_cost_fn=lambda e: e["weight"])
    shortest_paths = {
        (u, v): list(path)
        for u, paths_from_source in shortest_paths.items()
        for v, path in paths_from_source.items()
    }
    return shortest_paths


@memory.cache
def _get_all_pairs_path_lengths(graph: PyDiGraph) -> dict[tuple[int, int], int]:
    """Calculates and returns the node weighted shortest paths for all pairs.

    NOTE: output keys are waypoint key pairs!
    """
    logger.trace("get_all_pairs_path_lengths")

    shortest_paths = _get_all_pairs_shortest_paths(graph)
    shortest_paths = {
        key: sum(graph[w]["need_exploration_point"] for w in path) for key, path in shortest_paths.items()
    }
    shortest_paths.update({(n, n): 0 for n in graph.node_indices()})

    shortest_paths = {
        (graph[u]["waypoint_key"], graph[v]["waypoint_key"]): c for (u, v), c in shortest_paths.items()
    }
    return shortest_paths


@memory.cache
def _get_all_pairs_cartesian_distances(exploration_data: dict) -> dict[tuple[int, int], float]:
    """Calculates and returns the cartesian distances for all pairs.
    Cordinates are taken from the exploration data's x, y, z positions where x and z are
    the horizontal and vertical positions from center of the map.

    NOTE: output keys are waypoint key pairs!
    """
    import math

    logger.trace("get_all_pairs_cartesian_distances")

    return {
        (u, v): math.sqrt(
            (exploration_data[u]["position"]["x"] - exploration_data[v]["position"]["x"]) ** 2
            + (exploration_data[u]["position"]["z"] - exploration_data[v]["position"]["z"]) ** 2
        )
        for u in exploration_data.keys()
        for v in exploration_data.keys()
    }


# TODO: Research why the @memory.cache decorator doesn't work for this function
# most likely because it modifies the graph in place instead of returning a new graph
# @memory.cache
def _inject_super_root(G: PyDiGraph, super_root: dict[str, Any]):
    """Injects the superroot node into the graph using arcs between the super root
    and all nodes in its "link_list".

    NOTE: Flow for Super Root is always inbound. In terms of source, sink the terminals
    are the sources and the super root is the sink.
    """
    logger.debug("Injecting super root...")

    if "node_key_by_index" not in G.attrs:
        raise ValueError("'node_key_by_index' not in graph.attrs!")

    node_key_by_index = G.attrs["node_key_by_index"]

    if SUPER_ROOT in node_key_by_index.inv:
        logger.warning("  super root already exists in graph! Skipping injection...")
        return

    super_root_index = G.add_node(super_root)
    assert super_root_index not in node_key_by_index

    node_key_by_index[super_root_index] = SUPER_ROOT

    for node_key in super_root["link_list"]:
        node_index = node_key_by_index.inv[node_key]
        G.add_edge(node_index, super_root_index, super_root)

    G.attrs["node_key_by_index"] = node_key_by_index
    logger.debug(f"  injected at index {super_root_index}...")


def prune_NTD1(graph: PyDiGraph, non_removables: set[int] | None = None):
    """Prunes non terminal nodes of degree 1."""
    if non_removables is None:
        non_removables = set(
            graph.attrs.get("root_indices", [i for i in graph.node_indices() if graph[i]["is_town"]])
        )
        non_removables.update(
            set(
                graph.attrs.get(
                    "terminal_indices", [i for i in graph.node_indices() if graph[i]["is_terminal"]]
                )
            )
        )
        if graph.attrs.get("super_root_index", None) is not None:
            non_removables.add(graph.attrs["super_root_index"])
            non_removables.update(set(graph.attrs.get("super_terminal_indices", [])))

    num_removed = 0
    while removal_nodes := [
        v for v in graph.node_indices() if graph.out_degree(v) == 1 and v not in non_removables
    ]:
        graph.remove_nodes_from(removal_nodes)
        num_removed += len(removal_nodes)
        removal_nodes = []

    return num_removed
