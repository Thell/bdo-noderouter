# fuzz_suboptimal_viewer.py
"""
Visualizes mip and noderouter routing solutions on the BDO map for comparison.

NOTE: Sourcing this file directly starts the UI.
"""

import os
import re
import tempfile
import webbrowser
from copy import deepcopy
from enum import Enum

from branca.element import Element
import colorcet as cc
import flet as ft
from folium import FeatureGroup, Map, Marker, PolyLine, TileLayer
from folium.map import CustomPane
from folium.plugins import FeatureGroupSubGroup, GroupedLayerControl, BeautifyIcon

import rustworkx as rx
from loguru import logger
from rustworkx import PyDiGraph

import api_data_store as ds
from api_common import MAX_BUDGET, set_logger
from api_exploration_data import SUPER_ROOT, get_exploration_data, prune_NTD1
from api_rx_pydigraph import subgraph_stable, set_graph_terminal_sets_attribute
from orchestrator import execute_plan
from orchestrator_types import Plan, OptimizationFn, Instance
from orchestrator_pairing_strategy import PairingStrategy

from optimizer_mip import optimize_with_terminals as mip_optimize
from optimizer_nr import optimize_with_terminals_single as nr_optimize

TerminalsFG = tuple[FeatureGroup, dict[str, FeatureGroupSubGroup]]
GraphFG = tuple[FeatureGroup, FeatureGroup]

RE_TR_PAIRS = r"(\d+)\s*[:,]\s*(\d+)"

BASE_EDGE_COLOR = "cyan"
MIP_EDGE_COLOR = "yellow"
NR_EDGE_COLOR = "orange"

BASE_NODE_COLOR = "blue"
BASE_TOWN_NODE_COLOR = "red"
SUPER_ROOT_COLOR = "darkblue"
MIP_NODE_COLOR = "green"
NR_NODE_COLOR = "orange"

ROOT_CIRCLE_RADIUS = 2
BASE_NODE_RADIUS = 1
MIP_NODE_RADIUS = 2
NR_NODE_RADIUS = 2

BASE_EDGE_WEIGHT = 1
MIP_EDGE_WEIGHT = 6
NR_EDGE_WEIGHT = 3


class _GraphType(Enum):
    MAIN = 0
    MIP = 1
    NR = 2

    @property
    def edge_color(self):
        if self == _GraphType.MAIN:
            return BASE_EDGE_COLOR
        elif self == _GraphType.MIP:
            return MIP_EDGE_COLOR
        elif self == _GraphType.NR:
            return NR_EDGE_COLOR

    @property
    def edge_weight(self):
        if self == _GraphType.MAIN:
            return BASE_EDGE_WEIGHT
        elif self == _GraphType.MIP:
            return MIP_EDGE_WEIGHT
        elif self == _GraphType.NR:
            return NR_EDGE_WEIGHT

    @property
    def node_color(self):
        if self == _GraphType.MAIN:
            return BASE_NODE_COLOR
        elif self == _GraphType.MIP:
            return MIP_NODE_COLOR
        elif self == _GraphType.NR:
            return NR_NODE_COLOR

    @property
    def node_radius(self):
        if self == _GraphType.MAIN:
            return BASE_NODE_RADIUS
        elif self == _GraphType.MIP:
            return MIP_NODE_RADIUS
        elif self == _GraphType.NR:
            return NR_NODE_RADIUS

    @property
    def show_fg(self):
        if self == _GraphType.MAIN:
            return False
        else:
            return True


class _RootColor:
    colors: dict[int, str]
    suboptimal_roots: set[int]

    def __init__(self, mip_graph: PyDiGraph, nr_graph: PyDiGraph):
        terminal_sets = mip_graph.attrs["terminal_sets"]
        roots = sorted(list(terminal_sets))
        self.colors = {r: c for r, c in zip(roots, cc.b_glasbey_category10[: len(roots)])}

        # NOTE: Suboptimal roots are those in a different cc cluster in the nr_graph then the mip_graph.
        def get_root_groups(G: PyDiGraph) -> dict[int, frozenset[int]]:
            ccs = rx.strongly_connected_components(G)
            cc_roots = {i: {idx for idx in cc if idx in terminal_sets} for i, cc in enumerate(ccs)}
            return {root: frozenset(roots) for roots in cc_roots.values() for root in roots}

        mip_groups = get_root_groups(mip_graph)
        nr_groups = get_root_groups(nr_graph)
        self.suboptimal_roots = {r for r in mip_groups if mip_groups[r] != nr_groups[r]}

    def terminal_color(self, root_idx: int) -> str:
        return self.colors[root_idx]

    def is_suboptimal(self, root_idx: int) -> bool:
        return root_idx in self.suboptimal_roots


def _add_edges_from_graph(fg: FeatureGroup, G: PyDiGraph):
    """Add edges from the graph to the map."""
    coords = get_exploration_data().coords
    graph_type = G.attrs["graph_type"]
    color = graph_type.edge_color
    weight = graph_type.edge_weight

    for u_idx, v_idx in G.edge_list():
        u_key = G[u_idx]["waypoint_key"]
        v_key = G[v_idx]["waypoint_key"]

        # SAFETY: Edges are anti-parallel bi-directional edges with no self-loops
        #         except for SUPER_ROOT incident edges.
        if v_idx < u_idx and SUPER_ROOT not in [u_key, v_key]:
            PolyLine(
                locations=coords.as_geographics([u_idx, v_idx]),
                color=color,
                weight=weight,
                popup=f"Edge: {u_key} - {v_key}",
                tooltip=f"Edge: {u_key} - {v_key}",
            ).add_to(fg)


def _add_node_markers_from_graph(fg: FeatureGroup, G: PyDiGraph):
    coords = get_exploration_data().coords
    graph_type = G.attrs["graph_type"]

    for node_idx in G.node_indices():
        node = G[node_idx]
        key = node["waypoint_key"]
        cost = node["need_exploration_point"]

        popup_text = f"Node Key: {key}, Cost: {cost}"
        node_color = graph_type.node_color

        # NOTE: SUPER_ROOT is also a base town so it must be checked first
        if key == SUPER_ROOT:
            popup_text += " (Super Root)"
            node_color = SUPER_ROOT_COLOR
        elif node["is_base_town"]:
            node_color = BASE_TOWN_NODE_COLOR

        Marker(
            location=coords.as_geographic(node_idx),
            icon=BeautifyIcon(
                border_color=node_color,
                icon_size=[16, 16],
                icon_anchor=[8, 8],
                inner_icon_style="margin-top:-2px;",
                number=cost,
                text_color="black",
            ),  # type: ignore
            popup=popup_text,
            tooltip=popup_text,
        ).add_to(fg)


def _add_terminal_sets_markers(m: Map, G: PyDiGraph, root_color: _RootColor) -> TerminalsFG:
    """Add terminal sets using FeatureGroupSubGroup for hierarchical toggling."""

    def add_marker(type: str, location: tuple[float, float], key: int, cost: int, parent_fg):
        Marker(
            location=location,
            icon=BeautifyIcon(
                icon="house" if type == "Root" else "",
                icon_shape="marker",
                icon_size=[28, 28] if type == "Root" else [21, 21],
                icon_anchor=[14, 36] if type == "Root" else [11, 27],
                background_color=active_background_color,
                text_color="white",
                border_color=active_border_color,
            ),  # type: ignore
            popup=f"{type} Node Key: {key}, Cost: {cost}",
            tooltip=f"{type} Node Key: {key}, Cost: {cost}",
        ).add_to(parent_fg)

    fg_terminal_sets_master = FeatureGroup(name="All Terminal Sets", show=True)
    fg_terminal_sets_master.add_to(m)
    assert isinstance(fg_terminal_sets_master, FeatureGroup)
    terminal_set_feature_groups: dict[str, FeatureGroupSubGroup] = {}

    coords = get_exploration_data().coords
    terminal_sets = G.attrs["terminal_sets"]

    for root_idx in sorted(list(terminal_sets)):
        is_suboptimal = root_color.is_suboptimal(root_idx)
        root_key = G[root_idx]["waypoint_key"]
        terminal_set = terminal_sets[root_idx]

        layer_name = f"Terminal Set {root_key}"
        fg_terminal_set = FeatureGroupSubGroup(fg_terminal_sets_master, name=layer_name, show=is_suboptimal)
        fg_terminal_set.add_to(m)
        terminal_set_feature_groups[layer_name] = fg_terminal_set

        active_background_color = root_color.terminal_color(root_idx)
        active_border_color = "black" if is_suboptimal else "white"

        cost = G[root_idx]["need_exploration_point"]
        location = coords.as_geographic(root_idx)
        add_marker("Root", location, root_key, cost, fg_terminal_set)

        for terminal in terminal_set:
            cost = G[terminal]["need_exploration_point"]
            key = G[terminal]["waypoint_key"]
            location = coords.as_geographic(terminal)
            add_marker("Terminal", location, key, cost, fg_terminal_set)

    return fg_terminal_sets_master, terminal_set_feature_groups


def _add_graph_layer(m: Map, G: PyDiGraph) -> GraphFG:
    """Add graph nodes and edges feature groups to the map."""
    g_type = G.attrs["graph_type"]
    logger.debug(f"Setting up {g_type.name} graph layer...")

    fg_nodes = FeatureGroup(name=f"{g_type.name} Nodes", show=g_type.show_fg)
    fg_edges = FeatureGroup(name=f"{g_type.name} Edges", show=g_type.show_fg)
    _add_node_markers_from_graph(fg_nodes, G)
    _add_edges_from_graph(fg_edges, G)
    fg_nodes.add_to(m)
    fg_edges.add_to(m)

    return fg_nodes, fg_edges


def _visualize_solution_graphs(
    main_graph: PyDiGraph,
    mip_graph: PyDiGraph,
    mip_instance: Instance,
    nr_graph: PyDiGraph,
    nr_instance: Instance,
):
    m = Map(crs="Simple", location=[32.5, 0], zoom_start=2, zoom_snap=0.25, tiles=None)

    tile_pane = CustomPane("tile_pane", z_index=1)
    m.add_child(tile_pane)

    tile_layer = TileLayer(
        name="SubOptimal Solution Map",
        attr="Map Tiles @ BDO",
        min_zoom=1,
        max_zoom=7,
        no_wrap=True,
        pane="tile_pane",
        show=True,
        tiles=os.path.join(ds.path(), "maptiles", "{z}", "{x}_{y}.webp"),
    )
    tile_layer.add_to(m)

    fg_main_nodes, fg_main_edges = _add_graph_layer(m, main_graph)
    fg_mip_nodes, fg_mip_edges = _add_graph_layer(m, mip_graph)
    fg_nr_nodes, fg_nr_edges = _add_graph_layer(m, nr_graph)

    # NOTE: Terminal sets of roots and root colors of mip and nr are the same.
    logger.debug("Setting up terminal sets layer...")
    color_map = _RootColor(mip_graph, nr_graph)
    fg_terminal_master, terminal_groups = _add_terminal_sets_markers(m, mip_graph, color_map)

    m.keep_in_front(
        fg_main_edges,
        fg_mip_edges,
        fg_nr_edges,
        fg_main_nodes,
        fg_mip_nodes,
        fg_nr_nodes,
        fg_terminal_master,
        *terminal_groups.values(),
    )

    logger.debug("Setting up group layer controls...")
    group_layer_control = GroupedLayerControl(
        groups={
            "Base": [fg_main_nodes, fg_mip_nodes, fg_nr_nodes, fg_main_edges, fg_mip_edges, fg_nr_edges],
        },
        collapsed=False,
        exclusive_groups=False,
        position="topright",
    )
    group_layer_control.add_to(m)

    group_layer_control = GroupedLayerControl(
        groups={"Terminal Sets": [fg_terminal_master, *terminal_groups.values()]},
        collapsed=False,
        exclusive_groups=False,
        position="bottomright",
    )
    group_layer_control.add_to(m)

    logger.debug("Setting up graph and solution stats panel...")
    assert mip_instance.solution and nr_instance.solution
    stats_html = f"""
        <div style="position:absolute; z-index:100000; left:10px; top:10px;
                    background-color:#E0E0E0; padding:10px; border:1px solid black;">
            <h2>Stats</h2>
            <p>Main Nodes: {main_graph.num_nodes()}</p>
            <p>Main Edges: {main_graph.num_edges()}</p>
            <hr>
            <p>Roots: {mip_instance.terminals.roots}</p>
            <p>Terminals: {mip_instance.terminals.workers}</p>
            <p>Dangers: {mip_instance.terminals.dangers}</p>
            <hr>
            <p>MIP Nodes: {mip_instance.solution.num_nodes}</p>
            <p>MIP Edges: {mip_instance.solution.num_edges}</p>
            <p>MIP CC: {mip_instance.solution.num_components}</p>
            <p>MIP Cost: {mip_instance.solution.cost}</p>
            <hr>
            <p>NR Nodes: {nr_instance.solution.num_nodes}</p>
            <p>NR Edges: {nr_instance.solution.num_edges}</p>
            <p>NR CC: {nr_instance.solution.num_components}</p>
            <p>NR Cost: {nr_instance.solution.cost}</p>
        </div>
    """
    m.get_root().html.add_child(Element(stats_html))  # type: ignore

    tmp_file = os.path.join(tempfile.gettempdir(), "suboptimal_map.html")
    m.save(tmp_file)
    print(f"Map saved to {tmp_file}. Opening in default browser...")
    webbrowser.open("file://" + tmp_file)


def _visualize_instances(mip_instance: Instance, nr_instance: Instance):
    """Visualizes the mip and noderouter solutions on the BDO map."""

    assert mip_instance.solution and nr_instance.solution
    logger.info("=== Visualizing solutions ===")
    logger.info(f"mip: {mip_instance.solution.waypoints}")
    logger.info(f" nr: {nr_instance.solution.waypoints}")

    # NOTE: Technically we could use super graph for all instances but
    # the MIP problem would have more variables and take longer to solve
    # when the cache is not primed which is true for most custom plans.
    exploration_data = get_exploration_data()
    if mip_instance.terminals.dangers > 0:
        graph = exploration_data.super_graph
        # SAFETY: Super root is always the last node
        super_root = [graph.num_nodes() - 1]
    else:
        graph = exploration_data.graph
        super_root = []

    node_key_by_index = graph.attrs["node_key_by_index"]

    # SAFETY: deepcopy is required to avoid modifying the original graph upon attribute modification
    mip_solution_indices = super_root + [node_key_by_index.inv[i] for i in mip_instance.solution.waypoints]
    mip_graph = deepcopy(subgraph_stable(graph, mip_solution_indices))
    set_graph_terminal_sets_attribute(mip_graph, nr_instance.terminals.terminals)

    nr_solution_indices = super_root + [node_key_by_index.inv[i] for i in nr_instance.solution.waypoints]
    nr_graph = deepcopy(subgraph_stable(graph, nr_solution_indices))
    set_graph_terminal_sets_attribute(nr_graph, nr_instance.terminals.terminals)

    # NOTE: graph still has all leaf nodes in it and no leaf nodes need to be present other
    #       than the ones in the solution. This speeds up rendering and declutters the map.
    # SAFETY: copy() is shallow and deepcopy is required to avoid modifying the original graph upon node removal
    non_removables = set(mip_instance.solution.waypoints) | set(nr_instance.solution.waypoints)
    non_removables = {node_key_by_index.inv[i] for i in non_removables}
    main_graph = deepcopy(graph.copy())
    prune_NTD1(main_graph, non_removables)

    main_graph.attrs["graph_type"] = _GraphType.MAIN
    mip_graph.attrs["graph_type"] = _GraphType.MIP
    nr_graph.attrs["graph_type"] = _GraphType.NR

    _visualize_solution_graphs(main_graph, mip_graph, mip_instance, nr_graph, nr_instance)


####
# MARK: --- FLET UI ---
####
def _make_plan(
    plan_args: dict, optimization_fn: OptimizationFn, as_mip: bool = False, is_custom: bool = False
) -> Plan:
    allow_cache = True if as_mip else False
    strategy = PairingStrategy(plan_args["strategy"])

    # NOTE: If include_danger is True dangers are added to the plan yet custom strategy
    #       already defines them manually in terminal_pairs.
    include_danger = True if plan_args["dangers"] > 0 and not is_custom else False

    _plan = Plan(
        optimization_fn,
        ds.get_config("config"),
        plan_args["budget"],
        plan_args["percent"],
        plan_args["seed"],
        include_danger,
        strategy,
        allow_cache,
    )
    return _plan


def _process_custom_plan(terminal_pairs: dict[int, int]):
    ds.write_json("custom_strategy_terminal_pairs.json", terminal_pairs)

    exploration_data = get_exploration_data()

    danger_count = sum(1 for v in terminal_pairs.values() if v == SUPER_ROOT)
    plantzone_count = sum(1 for v in terminal_pairs.values() if v != SUPER_ROOT)
    percent = round(plantzone_count / exploration_data.max_plantzone_count * 100)

    plan_args = {
        "budget": MAX_BUDGET,  # Dummy value not presented
        "percent": percent,
        "seed": 0,
        "strategy": PairingStrategy.custom.value,
        "dangers": danger_count,
    }
    logger.debug(f"terminal_pairs: {terminal_pairs}")
    logger.debug(f"Plan args: {plan_args}")

    mip_plan = _make_plan(plan_args, mip_optimize, True, True)
    nr_plan = _make_plan(plan_args, nr_optimize, False, True)

    try:
        mip_instance = execute_plan(mip_plan)
        nr_instance = execute_plan(nr_plan)
        assert mip_instance.solution and nr_instance.solution
    except Exception as e:
        logger.error(f"Solving custom plan failed: {e}")
        raise e

    _visualize_instances(mip_instance, nr_instance)


def _process_selected_plan(plan_data):
    mip_plan = _make_plan(plan_data, mip_optimize, True, False)
    nr_plan = _make_plan(plan_data, nr_optimize, False, False)

    try:
        mip_instance = execute_plan(mip_plan)
        nr_instance = execute_plan(nr_plan)
        assert mip_instance.solution and nr_instance.solution
    except Exception as e:
        logger.error(f"Solving suboptimal plan failed: {e}")
        raise e

    _visualize_instances(mip_instance, nr_instance)


def _suboptimal_viewer_ui(page: ft.Page):
    def submit_custom(e):
        try:
            assert (input := custom_input.value)
            assert (data := {int(m[0]): int(m[1]) for m in re.findall(RE_TR_PAIRS, input)})
        except Exception:
            input_alert.open = True
            page.update()
            return

        try:
            _process_custom_plan(data)
        except Exception:
            processing_alert.open = True
        finally:
            page.update()

    def submit_selected(e):
        try:
            assert (data := e.control.data)
            _process_selected_plan(data)
            e.control.color = ft.Colors.with_opacity(0.2, "#00ff00")
            e.control.update()
        except Exception:
            processing_alert.open = True
        finally:
            page.update()

    rows = ds.read_json("worst_suboptimal_instances.json")
    data_table = ft.DataTable(columns=[ft.DataColumn(ft.Text(c)) for c in rows[0]], rows=[])
    for row in rows:
        cells = [ft.DataCell(ft.Text(v)) for v in row.values()]
        data_table.rows.append(ft.DataRow(cells=cells, data=row, on_select_changed=submit_selected))  # type: ignore

    custom_input = ft.TextField(multiline=True, min_lines=4, on_submit=submit_custom, shift_enter=True)
    custom_input.hint_text = "Custom Pairs - Examples:\n{t: r, ...}\n[(t, r), ...]\nt,r t,r ..."

    input_alert = ft.AlertDialog(content=ft.Text("Invalid or empty terminal,root pairs"))
    processing_alert = ft.AlertDialog(content=ft.Text("Unknown error occurred while executing plan"))
    suboptimal_title = ft.Text("SubOptimal Instances", size=16, weight=ft.FontWeight.BOLD)

    page.title = "Solution Visualizer"
    page.appbar = ft.AppBar(title=suboptimal_title, center_title=True)
    page.add(ft.Container(content=ft.Column([data_table], scroll=ft.ScrollMode.HIDDEN), expand=True))
    page.add(custom_input)
    page.overlay.extend([input_alert, processing_alert])
    page.window.width = 1620
    page.window.height = 810
    page.window.center()


def viewer_main():
    set_logger(ds.get_config("config"))
    ft.app(target=_suboptimal_viewer_ui)


if __name__ == "__main__":
    viewer_main()
