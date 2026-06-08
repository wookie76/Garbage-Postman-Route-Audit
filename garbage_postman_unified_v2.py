#!/usr/bin/env python3
"""Project-grade Chinese Postman / garbage-truck route solver.

Rebuilds the Green Valley garbage-route example as a reproducible Python project.

Core algorithm:
    1. Build weighted undirected street graph.
    2. Find odd-degree vertices.
    3. Compute shortest paths between odd vertices.
    4. Solve minimum-weight perfect matching over odd vertices.
    5. Duplicate selected shortest paths.
    6. Verify Eulerian multigraph.
    7. Extract closed Eulerian route from depot.
    8. Save human-readable visual diagnostics.

Why this is not TSP:
    Chinese Postman traverses every edge/street at least once.
    TSP visits every node/customer once.

Run:
    python garbage_postman_unified_v2.py solve --out-dir garbage_postman_outputs_v2
    python garbage_postman_unified_v2.py validate
    python garbage_postman_unified_v2.py fab-map

Dependencies:
    pip install networkx matplotlib seaborn typer rich pydantic loguru tqdm
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import typer
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm.auto import tqdm

Node = str
Pair = tuple[Node, Node]
Route = list[Node]
SolverName = Literal["networkx"]

APP = typer.Typer(
    add_completion=False,
    help="Solve and visualize the Green Valley Chinese Postman route.",
)
CONSOLE = Console()


@dataclass(frozen=True)
class PlotStyle:
    """Centralized visual design tokens.

    Uses Matplotlib Agg for headless output and Seaborn's icefire palette for a
    restrained diverging scheme. Palette is applied consistently:
      - dark blue: original/first-pass route
      - warm red: repeated/duplicate traversal
      - pale blue: normal nodes
      - gold: depot
      - warm odd-node color: odd-degree imbalance points
    """

    original_edge: str
    repeat_edge: str
    node: str
    depot: str
    odd_node: str
    euler_node: str
    text: str
    muted: str
    background: str
    panel: str
    grid: str
    edge_shadow: str
    figure_size: tuple[float, float] = (16.0, 9.0)
    dpi: int = 180


@dataclass(frozen=True)
class PostmanSolution:
    """Solved Chinese Postman route and audit artifacts."""

    base_cost: float
    added_cost: float
    total_cost: float
    odd_nodes: list[Node]
    pair_distances: dict[Pair, float]
    pair_paths: dict[Pair, Route]
    optimal_pairs: list[Pair]
    duplicate_edges: list[Pair]
    duplicate_paths: dict[Pair, Route]
    route: Route
    circuit_edges: list[Pair]


class EdgeSpec(BaseModel):
    """Validated undirected weighted edge."""

    model_config = ConfigDict(frozen=True)

    u: str = Field(min_length=1)
    v: str = Field(min_length=1)
    weight: PositiveFloat

    @field_validator("u", "v")
    @classmethod
    def normalize_node(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Node name cannot be blank")
        return cleaned

    @field_validator("v")
    @classmethod
    def prevent_self_loop(cls, value: str, info) -> str:  # type: ignore[no-untyped-def]
        u = info.data.get("u")
        if u is not None and value == u:
            raise ValueError("Self-loops are not supported in this demo graph")
        return value


class AppConfig(BaseModel):
    """Validated CLI configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    out_dir: Path = Path("garbage_postman_outputs_v2")
    depot: str = Field(default="D", min_length=1)
    solver: SolverName = "networkx"
    write_plots: bool = True
    progress: bool = True
    strict: bool = True
    log_file: bool = True

    @field_validator("depot")
    @classmethod
    def normalize_depot(cls, value: str) -> str:
        return value.strip().upper()


class RouteValidationError(RuntimeError):
    """Raised when solved route fails correctness checks."""


POSITIONS: dict[Node, tuple[float, float]] = {
    "A": (0.0, 3.0),
    "B": (2.0, 3.0),
    "C": (4.0, 3.0),
    "F": (6.0, 3.0),
    "D": (2.0, 1.5),
    "E": (4.0, 1.5),
    "G": (0.0, 0.6),
    "H": (2.0, 0.0),
    "J": (4.0, 0.0),
    "I": (6.0, 1.0),
}

STREET_EDGES: tuple[EdgeSpec, ...] = tuple(
    EdgeSpec(u=u, v=v, weight=w)
    for u, v, w in [
        ("A", "B", 3),
        ("B", "C", 4),
        ("C", "F", 6),
        ("F", "I", 4),
        ("I", "J", 4),
        ("J", "H", 6),
        ("H", "G", 3),
        ("G", "D", 4),
        ("D", "B", 5),
        ("D", "H", 7),
        ("D", "E", 4),
        ("E", "B", 2),
        ("E", "H", 5),
        ("E", "I", 3),
        ("D", "J", 5),
        ("G", "J", 5),
    ]
)

FAB_ANALOGY_ROWS: tuple[tuple[str, str], ...] = (
    ("street edge", "process-step transition"),
    ("travel time", "cycle time, queue time, cost, or risk"),
    ("depot", "lot release, route anchor, or finish anchor"),
    ("odd node", "flow imbalance point that prevents one-pass closed route"),
    ("duplicated edge", "repeat move, rework, inspection, or metrology revisit"),
    (
        "Eulerian graph",
        "route where every transition can be audited once in closed walk",
    ),
    ("closed postman route", "complete audit traversal with minimum repeat burden"),
)


def make_style() -> PlotStyle:
    """Create clear, color-stable plotting style from seaborn icefire palette."""

    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "figure.facecolor": "#f8f7f4",
            "axes.facecolor": "#fbfaf7",
            "axes.edgecolor": "#2d2d2d",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.color": "#d8d6d0",
            "grid.linewidth": 0.9,
            "font.family": "DejaVu Sans",
        },
    )
    palette = sns.color_palette("icefire", n_colors=8).as_hex()
    return PlotStyle(
        original_edge=palette[1],
        repeat_edge=palette[6],
        node="#9fd3e6",
        depot="#ffd166",
        odd_node="#f28482",
        euler_node="#b7e4c7",
        text="#1f2933",
        muted="#6b7280",
        background="#f8f7f4",
        panel="#fbfaf7",
        grid="#d8d6d0",
        edge_shadow="#363636",
    )


def iter_progress(items: Iterable, enabled: bool, desc: str) -> Iterable:
    """Wrap iterable in tqdm only when requested."""

    if not enabled:
        return items
    return tqdm(items, desc=desc, leave=False)


def configure_logging(config: AppConfig) -> None:
    """Configure loguru with concise console logging and optional file logging."""

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    if config.log_file:
        log_dir = config.out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "garbage_postman_{time:YYYYMMDD_HHmmss}.log",
            level="DEBUG",
            rotation="1 MB",
            retention=5,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )


def canonical_edge(u: Node, v: Node) -> Pair:
    """Return stable undirected edge key."""

    return tuple(sorted((u, v)))  # type: ignore[return-value]


def build_graph(edges: Iterable[EdgeSpec]) -> nx.Graph:
    """Build weighted undirected graph from edge specs."""

    graph = nx.Graph()
    seen: set[Pair] = set()
    for edge in edges:
        key = canonical_edge(edge.u, edge.v)
        if key in seen:
            raise ValueError(f"Duplicate original edge not allowed: {edge.u}-{edge.v}")
        seen.add(key)
        graph.add_edge(edge.u, edge.v, weight=float(edge.weight))

    missing_positions = sorted(set(graph.nodes) - set(POSITIONS))
    if missing_positions:
        raise ValueError(f"Missing deterministic layout positions: {missing_positions}")
    if not nx.is_connected(graph):
        raise ValueError("Chinese Postman solver requires a connected graph")
    return graph


def graph_base_cost(graph: nx.Graph) -> float:
    """Sum each original undirected edge weight once."""

    return sum(float(data["weight"]) for _, _, data in graph.edges(data=True))


def find_odd_nodes(graph: nx.Graph | nx.MultiGraph) -> list[Node]:
    """Find vertices with odd degree."""

    return sorted(node for node, degree in graph.degree() if degree % 2 == 1)


def shortest_path_lookup(
    graph: nx.Graph,
    odd_nodes: list[Node],
    progress: bool,
) -> tuple[dict[Pair, float], dict[Pair, Route]]:
    """Compute weighted shortest-path distances and node paths for odd-node pairs."""

    pairs = list(itertools.combinations(odd_nodes, 2))
    distances: dict[Pair, float] = {}
    paths: dict[Pair, Route] = {}
    for u, v in iter_progress(pairs, progress, "Odd-pair shortest paths"):
        key = canonical_edge(u, v)
        distances[key] = float(nx.shortest_path_length(graph, u, v, weight="weight"))
        paths[key] = list(nx.shortest_path(graph, u, v, weight="weight"))
    return distances, paths


def solve_minimum_pairing(
    odd_nodes: list[Node], pair_distances: dict[Pair, float]
) -> list[Pair]:
    """Find minimum-cost perfect matching among odd-degree nodes."""

    if len(odd_nodes) % 2 != 0:
        raise ValueError("Odd-degree vertex count must be even by Handshaking Lemma")
    if not odd_nodes:
        return []

    complete = nx.Graph()
    complete.add_nodes_from(odd_nodes)
    for (u, v), distance in pair_distances.items():
        complete.add_edge(u, v, weight=distance)

    matching = nx.algorithms.matching.min_weight_matching(complete, weight="weight")
    return sorted(canonical_edge(u, v) for u, v in matching)


def duplicate_path_edges(
    graph: nx.Graph,
    pairs: list[Pair],
    pair_paths: dict[Pair, Route],
    progress: bool,
) -> tuple[nx.MultiGraph, list[Pair], dict[Pair, Route]]:
    """Add duplicate edges along selected shortest paths to create Eulerian multigraph."""

    eulerized = nx.MultiGraph()
    for u, v, data in graph.edges(data=True):
        eulerized.add_edge(u, v, weight=float(data["weight"]), kind="original")

    duplicate_edges: list[Pair] = []
    selected_paths: dict[Pair, Route] = {}
    for pair in iter_progress(pairs, progress, "Duplicating shortest paths"):
        path = pair_paths[pair]
        selected_paths[pair] = path
        for u, v in itertools.pairwise(path):
            weight = float(graph[u][v]["weight"])
            eulerized.add_edge(u, v, weight=weight, kind="duplicate")
            duplicate_edges.append(canonical_edge(u, v))

    return eulerized, duplicate_edges, selected_paths


def multigraph_route_cost(graph: nx.MultiGraph, circuit_edges: list[Pair]) -> float:
    """Compute route cost from circuit edge sequence, consuming parallel edges once."""

    remaining: dict[Pair, list[float]] = {}
    for u, v, data in graph.edges(data=True):
        remaining.setdefault(canonical_edge(u, v), []).append(float(data["weight"]))

    total = 0.0
    for u, v in circuit_edges:
        key = canonical_edge(u, v)
        if key not in remaining or not remaining[key]:
            raise RouteValidationError(f"Circuit uses unavailable edge {u}-{v}")
        total += remaining[key].pop()
    return total


def solve_chinese_postman(
    graph: nx.Graph, config: AppConfig
) -> tuple[PostmanSolution, nx.MultiGraph]:
    """Solve undirected Chinese Postman Problem for weighted connected graph."""

    if config.depot not in graph:
        raise ValueError(f"Depot {config.depot!r} not found in graph")
    if config.solver != "networkx":
        raise ValueError(
            "Only solver='networkx' is implemented. PuLP reserved for constrained MILP extension."
        )

    logger.info("Computing graph audit")
    base_cost = graph_base_cost(graph)
    odd_nodes = find_odd_nodes(graph)
    logger.info("Odd-degree nodes: {}", ", ".join(odd_nodes) or "none")

    pair_distances, pair_paths = shortest_path_lookup(
        graph, odd_nodes, progress=config.progress
    )
    optimal_pairs = solve_minimum_pairing(odd_nodes, pair_distances)
    added_cost = sum(pair_distances[pair] for pair in optimal_pairs)
    logger.info("Optimal odd-node pairs: {}", optimal_pairs)

    eulerized, duplicate_edges, duplicate_paths = duplicate_path_edges(
        graph=graph,
        pairs=optimal_pairs,
        pair_paths=pair_paths,
        progress=config.progress,
    )

    final_odds = find_odd_nodes(eulerized)
    if final_odds:
        raise RouteValidationError(f"Eulerized graph still has odd nodes: {final_odds}")
    if not nx.is_eulerian(eulerized):
        raise RouteValidationError("Eulerized graph is not Eulerian")

    circuit_edges = [
        (u, v) for u, v in nx.eulerian_circuit(eulerized, source=config.depot)
    ]
    route = [config.depot]
    route.extend(v for _, v in circuit_edges)
    total_cost = multigraph_route_cost(eulerized, circuit_edges)

    solution = PostmanSolution(
        base_cost=base_cost,
        added_cost=added_cost,
        total_cost=total_cost,
        odd_nodes=odd_nodes,
        pair_distances=pair_distances,
        pair_paths=pair_paths,
        optimal_pairs=optimal_pairs,
        duplicate_edges=duplicate_edges,
        duplicate_paths=duplicate_paths,
        route=route,
        circuit_edges=circuit_edges,
    )
    validate_solution(graph, eulerized, solution, depot=config.depot)
    return solution, eulerized


def validate_solution(
    original: nx.Graph,
    eulerized: nx.MultiGraph,
    solution: PostmanSolution,
    depot: Node,
) -> None:
    """Validate closed route, coverage, costs, and Eulerian conversion."""

    if solution.route[0] != depot or solution.route[-1] != depot:
        raise RouteValidationError("Route must start and end at depot")

    covered_original = {canonical_edge(u, v) for u, v in solution.circuit_edges}
    required_original = {canonical_edge(u, v) for u, v in original.edges()}
    missing = required_original - covered_original
    if missing:
        raise RouteValidationError(f"Route missed original edges: {sorted(missing)}")

    if len(solution.circuit_edges) != eulerized.number_of_edges():
        raise RouteValidationError(
            "Eulerian circuit did not consume every multigraph edge"
        )

    expected = solution.base_cost + solution.added_cost
    if not math.isclose(solution.total_cost, expected, rel_tol=0, abs_tol=1e-9):
        raise RouteValidationError(
            f"Cost mismatch: route={solution.total_cost}, base+added={expected}"
        )


def fmt_cost(value: float) -> str:
    """Format numeric minutes cleanly."""

    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def edge_weight_labels(graph: nx.Graph) -> dict[Pair, str]:
    """Build edge-label dict for NetworkX drawing."""

    return {
        (u, v): fmt_cost(float(data["weight"])) for u, v, data in graph.edges(data=True)
    }


def setup_axis(
    ax: plt.Axes, title: str, subtitle: str | None, style: PlotStyle
) -> None:
    """Apply map-like layout: clean frame, fixed aspect, readable title."""

    ax.set_title(title, fontsize=22, fontweight="bold", color=style.text, pad=24)
    if subtitle:
        ax.text(
            0.5,
            1.005,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            color=style.muted,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.margins(0.14)


def draw_edge_labels(graph: nx.Graph, ax: plt.Axes, style: PlotStyle) -> None:
    """Draw weight labels with white halo boxes for cartographic readability."""

    labels = nx.draw_networkx_edge_labels(
        graph,
        POSITIONS,
        edge_labels=edge_weight_labels(graph),
        ax=ax,
        font_size=10,
        font_color=style.text,
        bbox={
            "boxstyle": "round,pad=0.16",
            "fc": style.panel,
            "ec": "none",
            "alpha": 0.82,
        },
    )
    for label in labels.values():
        label.set_zorder(6)


def draw_nodes(
    graph: nx.Graph,
    ax: plt.Axes,
    style: PlotStyle,
    depot: Node,
    odd_nodes: set[Node] | None = None,
    eulerized: bool = False,
) -> None:
    """Draw nodes with semantically stable colors."""

    odd_nodes = odd_nodes or set()
    colors: list[str] = []
    for node in graph.nodes:
        if node == depot:
            colors.append(style.depot)
        elif node in odd_nodes:
            colors.append(style.odd_node)
        elif eulerized:
            colors.append(style.euler_node)
        else:
            colors.append(style.node)

    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        ax=ax,
        node_color=colors,
        node_size=1120,
        edgecolors=style.text,
        linewidths=1.4,
    )
    nx.draw_networkx_labels(
        graph,
        POSITIONS,
        ax=ax,
        font_size=14,
        font_weight="bold",
        font_color=style.text,
    )


def add_legend_box(ax: plt.Axes, lines: list[str], style: PlotStyle) -> None:
    """Add compact explanatory legend text."""

    text = "\n".join(lines)
    ax.text(
        0.015,
        0.025,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=style.text,
        bbox={
            "boxstyle": "round,pad=0.45",
            "fc": style.panel,
            "ec": style.grid,
            "alpha": 0.94,
        },
        zorder=10,
    )


def draw_base_graph(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_path: Path,
    style: PlotStyle,
    depot: Node,
) -> None:
    """Save original graph with odd nodes and depot highlighted."""

    fig, ax = plt.subplots(figsize=style.figure_size)
    fig.patch.set_facecolor(style.background)
    ax.set_facecolor(style.panel)

    nx.draw_networkx_edges(
        graph,
        POSITIONS,
        ax=ax,
        width=2.4,
        alpha=0.82,
        edge_color=style.edge_shadow,
    )
    draw_edge_labels(graph, ax, style)
    draw_nodes(graph, ax, style, depot=depot, odd_nodes=set(solution.odd_nodes))
    setup_axis(
        ax,
        "Original Street Network",
        "Depot D in gold. Odd-degree vertices in warm red. Edge labels are minutes.",
        style,
    )
    add_legend_box(
        ax,
        [
            f"Nodes: {graph.number_of_nodes()} | Streets: {graph.number_of_edges()}",
            f"Base cost: {fmt_cost(solution.base_cost)} min",
            f"Odd nodes: {', '.join(solution.odd_nodes)}",
        ],
        style,
    )
    fig.tight_layout()
    fig.savefig(
        out_path, dpi=style.dpi, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def draw_pairing_graph(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_path: Path,
    style: PlotStyle,
    depot: Node,
) -> None:
    """Save graph with optimal odd-node duplicate paths highlighted."""

    fig, ax = plt.subplots(figsize=style.figure_size)
    fig.patch.set_facecolor(style.background)
    ax.set_facecolor(style.panel)

    nx.draw_networkx_edges(
        graph,
        POSITIONS,
        ax=ax,
        width=1.8,
        alpha=0.30,
        edge_color=style.edge_shadow,
    )

    duplicate_counts = Counter(solution.duplicate_edges)
    for index, ((u, v), count) in enumerate(sorted(duplicate_counts.items())):
        radius = 0.07 + 0.04 * index
        nx.draw_networkx_edges(
            graph,
            POSITIONS,
            edgelist=[(u, v)],
            ax=ax,
            width=4.2 + count * 0.4,
            edge_color=style.repeat_edge,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            connectionstyle=f"arc3,rad={radius}",
            min_source_margin=18,
            min_target_margin=18,
        )

    draw_edge_labels(graph, ax, style)
    draw_nodes(graph, ax, style, depot=depot, odd_nodes=set(solution.odd_nodes))
    pair_text = " | ".join(
        f"{u}-{v}: {'→'.join(solution.duplicate_paths[(u, v)])}"
        for u, v in solution.optimal_pairs
    )
    setup_axis(
        ax,
        "Minimum Added-Cost Pairing",
        pair_text,
        style,
    )
    add_legend_box(
        ax,
        [
            "Warm arcs = streets duplicated by shortest odd-node pairing",
            f"Added repeat cost: {fmt_cost(solution.added_cost)} min",
            "Method: NetworkX minimum-weight matching",
        ],
        style,
    )
    fig.tight_layout()
    fig.savefig(
        out_path, dpi=style.dpi, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def draw_eulerized_graph(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_path: Path,
    style: PlotStyle,
    depot: Node,
) -> None:
    """Save original graph plus duplicated edges as curved overlays."""

    fig, ax = plt.subplots(figsize=style.figure_size)
    fig.patch.set_facecolor(style.background)
    ax.set_facecolor(style.panel)

    nx.draw_networkx_edges(
        graph,
        POSITIONS,
        ax=ax,
        width=2.0,
        alpha=0.56,
        edge_color=style.edge_shadow,
    )
    draw_edge_labels(graph, ax, style)

    duplicate_counts = Counter(solution.duplicate_edges)
    for index, ((u, v), count) in enumerate(sorted(duplicate_counts.items())):
        radius = 0.15 + 0.045 * index
        nx.draw_networkx_edges(
            graph,
            POSITIONS,
            edgelist=[(u, v)],
            ax=ax,
            width=3.3 + count * 0.35,
            edge_color=style.repeat_edge,
            connectionstyle=f"arc3,rad={radius}",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            min_source_margin=18,
            min_target_margin=18,
        )

    draw_nodes(graph, ax, style, depot=depot, eulerized=True)
    setup_axis(
        ax,
        "Eulerized Graph",
        "All vertices now have even degree. Curved warm arcs show repeated streets.",
        style,
    )
    add_legend_box(
        ax,
        [
            "Goal: convert non-Eulerian graph into Eulerian multigraph",
            "Even degrees allow closed tour over every multigraph edge once",
            f"Total cost after repeats: {fmt_cost(solution.total_cost)} min",
        ],
        style,
    )
    fig.tight_layout()
    fig.savefig(
        out_path, dpi=style.dpi, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def draw_final_route(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_path: Path,
    style: PlotStyle,
    depot: Node,
) -> None:
    """Save final closed tour with directed arrows and step labels."""

    fig, ax = plt.subplots(figsize=style.figure_size)
    fig.patch.set_facecolor(style.background)
    ax.set_facecolor(style.panel)

    nx.draw_networkx_edges(
        graph,
        POSITIONS,
        ax=ax,
        width=1.4,
        alpha=0.22,
        edge_color=style.edge_shadow,
    )
    draw_edge_labels(graph, ax, style)
    draw_nodes(graph, ax, style, depot=depot)

    parallel_seen: Counter[Pair] = Counter()
    for step, (u, v) in enumerate(solution.circuit_edges, start=1):
        key = canonical_edge(u, v)
        parallel_seen[key] += 1
        seen = parallel_seen[key]
        radius = 0.075 * ((seen + 1) // 2)
        if seen % 2 == 0:
            radius *= -1
        color = style.repeat_edge if seen > 1 else style.original_edge
        nx.draw_networkx_edges(
            graph,
            POSITIONS,
            edgelist=[(u, v)],
            ax=ax,
            width=2.35,
            edge_color=color,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            connectionstyle=f"arc3,rad={radius}",
            min_source_margin=23,
            min_target_margin=23,
        )
        x1, y1 = POSITIONS[u]
        x2, y2 = POSITIONS[v]
        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2 + radius * 0.72
        ax.text(
            label_x,
            label_y,
            str(step),
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            color=style.text,
            bbox={
                "boxstyle": "circle,pad=0.23",
                "fc": style.panel,
                "ec": color,
                "lw": 1.2,
                "alpha": 0.92,
            },
            zorder=9,
        )

    route_text = " → ".join(solution.route)
    wrapped = route_text if len(route_text) <= 150 else route_text[:147] + "..."
    setup_axis(
        ax,
        f"Final Closed Tour: {fmt_cost(solution.total_cost)} Minutes",
        "Dark blue = first traversal. Warm red = repeated traversal. Circles show route step order.",
        style,
    )
    ax.text(
        0.5,
        -0.055,
        wrapped,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=style.text,
    )
    add_legend_box(
        ax,
        [
            f"Start/end depot: {depot}",
            f"Route edges traversed: {len(solution.circuit_edges)}",
            "Validation: every original street covered at least once",
        ],
        style,
    )
    fig.tight_layout()
    fig.savefig(
        out_path, dpi=style.dpi, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def draw_cost_breakdown(
    solution: PostmanSolution, out_path: Path, style: PlotStyle
) -> None:
    """Save base/additional/total cost bar chart."""

    fig, ax = plt.subplots(figsize=style.figure_size)
    fig.patch.set_facecolor(style.background)
    ax.set_facecolor(style.panel)

    labels = [
        "Base\nall streets once",
        "Added\nrepeated streets",
        "Total\nclosed route",
    ]
    values = [solution.base_cost, solution.added_cost, solution.total_cost]
    colors = [style.original_edge, style.repeat_edge, style.depot]
    bars = ax.bar(labels, values, color=colors, edgecolor=style.text, linewidth=1.1)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{fmt_cost(value)} min",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=style.text,
        )
    ax.set_ylabel("Minutes", fontsize=13, color=style.text)
    ax.set_title(
        "Chinese Postman Cost Breakdown",
        fontsize=22,
        fontweight="bold",
        color=style.text,
        pad=20,
    )
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.36)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        out_path, dpi=style.dpi, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def save_visualizations(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_dir: Path,
    config: AppConfig,
) -> list[Path]:
    """Save all plot artifacts."""

    style = make_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "01_original_graph.png",
        out_dir / "02_odd_pair_shortest_paths.png",
        out_dir / "03_eulerized_graph.png",
        out_dir / "04_final_route.png",
        out_dir / "05_cost_breakdown.png",
    ]
    plot_jobs = [
        (draw_base_graph, paths[0]),
        (draw_pairing_graph, paths[1]),
        (draw_eulerized_graph, paths[2]),
        (draw_final_route, paths[3]),
    ]
    for draw_func, path in iter_progress(
        plot_jobs, config.progress, "Saving route maps"
    ):
        draw_func(graph, solution, path, style, config.depot)
    draw_cost_breakdown(solution, paths[4], style)
    return paths


def build_graph_audit_table(
    graph: nx.Graph, solution: PostmanSolution, depot: Node
) -> Table:
    """Create Rich graph-audit metrics table."""

    table = Table(title="Graph Audit", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Nodes", str(graph.number_of_nodes()))
    table.add_row("Original streets", str(graph.number_of_edges()))
    table.add_row("Depot", depot)
    table.add_row("Base edge cost", f"{fmt_cost(solution.base_cost)} minutes")
    table.add_row("Odd-degree nodes", ", ".join(solution.odd_nodes) or "none")
    table.add_row("Eulerian before fix", "yes" if not solution.odd_nodes else "no")
    return table


def build_degree_table(graph: nx.Graph) -> Table:
    """Create Rich node-degree table."""

    table = Table(title="Node Degrees", show_header=True, header_style="bold cyan")
    table.add_column("Node", style="bold")
    table.add_column("Degree", justify="right")
    table.add_column("Status")
    for node, degree in sorted(graph.degree()):
        table.add_row(node, str(degree), "odd" if degree % 2 else "even")
    return table


def build_matching_table(solution: PostmanSolution) -> Table:
    """Create Rich matching / duplicated path table."""

    table = Table(
        title="Minimum Added-Cost Matching", show_header=True, header_style="bold cyan"
    )
    table.add_column("Pair", style="bold")
    table.add_column("Shortest path")
    table.add_column("Added cost", justify="right")
    for pair in solution.optimal_pairs:
        path = solution.duplicate_paths[pair]
        table.add_row(
            f"{pair[0]}-{pair[1]}",
            " → ".join(path),
            f"{fmt_cost(solution.pair_distances[pair])} min",
        )
    table.add_section()
    table.add_row("Total", "", f"{fmt_cost(solution.added_cost)} min")
    return table


def build_result_table(solution: PostmanSolution, depot: Node) -> Table:
    """Create Rich final result table."""

    table = Table(
        title="Final Route Metrics", show_header=True, header_style="bold cyan"
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Base cost", f"{fmt_cost(solution.base_cost)} minutes")
    table.add_row("Duplicate cost", f"{fmt_cost(solution.added_cost)} minutes")
    table.add_row("Total route cost", f"{fmt_cost(solution.total_cost)} minutes")
    table.add_row("Route starts at depot", str(solution.route[0] == depot))
    table.add_row("Route ends at depot", str(solution.route[-1] == depot))
    table.add_row("Closed route", " → ".join(solution.route))
    return table


def build_fab_table() -> Table:
    """Create Rich fab-routing analogy table."""

    table = Table(
        title="Fab / Wafer-Route Analogy", show_header=True, header_style="bold cyan"
    )
    table.add_column("Routing concept", style="bold")
    table.add_column("Fab interpretation")
    for left, right in FAB_ANALOGY_ROWS:
        table.add_row(left, right)
    return table


def print_reports(
    graph: nx.Graph, solution: PostmanSolution, config: AppConfig, saved: list[Path]
) -> None:
    """Print solver reports with Rich tables."""

    CONSOLE.print(build_graph_audit_table(graph, solution, config.depot))
    CONSOLE.print(build_degree_table(graph))
    CONSOLE.print(build_matching_table(solution))
    CONSOLE.print(build_result_table(solution, config.depot))
    CONSOLE.print(build_fab_table())

    saved_table = Table(
        title="Saved Visualizations", show_header=True, header_style="bold cyan"
    )
    saved_table.add_column("File")
    for path in saved:
        saved_table.add_row(str(path))
    CONSOLE.print(saved_table)


def run_solution(
    config: AppConfig,
) -> tuple[nx.Graph, nx.MultiGraph, PostmanSolution, list[Path]]:
    """Run full solve pipeline."""

    configure_logging(config)
    logger.info("Building graph")
    graph = build_graph(STREET_EDGES)
    solution, eulerized = solve_chinese_postman(graph, config)
    saved: list[Path] = []
    if config.write_plots:
        logger.info("Saving visualizations to {}", config.out_dir)
        saved = save_visualizations(graph, solution, config.out_dir, config)
    logger.info("Done. Total route cost: {} minutes", fmt_cost(solution.total_cost))
    return graph, eulerized, solution, saved


def handle_error(exc: Exception) -> None:
    """Render user-facing error and exit nonzero."""

    logger.exception("Run failed")
    CONSOLE.print(
        Panel.fit(
            f"[bold red]Run failed[/bold red]\n{type(exc).__name__}: {exc}",
            border_style="red",
        )
    )
    raise typer.Exit(code=1) from exc


@APP.command()
def solve(
    out_dir: Path = typer.Option(
        Path("garbage_postman_outputs_v2"),
        "--out-dir",
        help="Directory where PNG visualizations and logs will be saved.",
    ),
    depot: str = typer.Option("D", "--depot", help="Start/end node for closed route."),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show tqdm progress bars."
    ),
    write_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Save PNG visualizations."
    ),
    log_file: bool = typer.Option(
        True, "--log-file/--no-log-file", help="Write log file under out-dir/logs."
    ),
) -> None:
    """Solve article graph, validate route, print metrics, save 16:9 maps."""

    config = AppConfig(
        out_dir=out_dir,
        depot=depot,
        progress=progress,
        write_plots=write_plots,
        log_file=log_file,
    )
    try:
        graph, _, solution, saved = run_solution(config)
        print_reports(graph, solution, config, saved)
    except Exception as exc:  # noqa: BLE001 - CLI boundary should render all failures.
        handle_error(exc)


@APP.command()
def validate(
    depot: str = typer.Option("D", "--depot", help="Start/end node for closed route."),
) -> None:
    """Run smoke test against expected article result: 70 + 12 = 82."""

    config = AppConfig(
        out_dir=Path("garbage_postman_outputs_v2"),
        depot=depot,
        progress=False,
        write_plots=False,
        log_file=False,
    )
    try:
        graph, _, solution, _ = run_solution(config)
        assert graph.number_of_nodes() == 10
        assert graph.number_of_edges() == 16
        assert solution.odd_nodes == ["A", "D", "G", "I"]
        assert math.isclose(solution.base_cost, 70.0)
        assert math.isclose(solution.added_cost, 12.0)
        assert math.isclose(solution.total_cost, 82.0)
        CONSOLE.print(
            Panel.fit(
                "[bold green]Validation passed[/bold green]: 70 + 12 = 82",
                border_style="green",
            )
        )
    except Exception as exc:  # noqa: BLE001
        handle_error(exc)


@APP.command("fab-map")
def fab_map() -> None:
    """Print fab / wafer-route analogy table only."""

    CONSOLE.print(build_fab_table())
    CONSOLE.print(
        Panel.fit(
            "Use exact graph algorithms where assumptions hold; use guarded heuristics "
            "when fab constraints make pure optimality less useful than auditable near-optimal routes.",
            title="Portfolio stance",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    APP()
