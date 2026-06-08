#!/usr/bin/env python3
"""Unified Chinese Postman / garbage truck route solver with visual diagnostics.

Rebuilds the Green Valley garbage-route example as a reproducible Python project:
- Builds a weighted undirected street graph.
- Solves the undirected Chinese Postman Problem.
- Verifies route correctness.
- Saves deterministic visualizations.
- Prints a manufacturing/fab-routing analogy table.

Run:
    python garbage_postman_unified.py --out-dir outputs

Dependencies:
    pip install networkx matplotlib
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

Node = str
Edge = tuple[Node, Node, float]
Pair = tuple[Node, Node]
Route = list[Node]


@dataclass(frozen=True)
class PostmanSolution:
    """Solved Chinese Postman route and audit artifacts."""

    base_cost: float
    added_cost: float
    total_cost: float
    odd_nodes: list[Node]
    optimal_pairs: list[Pair]
    duplicate_edges: list[Pair]
    duplicate_paths: dict[Pair, Route]
    route: Route
    circuit_edges: list[Pair]


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


STREET_EDGES: list[Edge] = [
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
    (
        "closed postman route",
        "complete audit traversal of all transitions with minimum repeat burden",
    ),
)


class RouteValidationError(RuntimeError):
    """Raised when solved route fails correctness checks."""


def canonical_edge(u: Node, v: Node) -> Pair:
    """Return stable undirected edge key."""

    return tuple(sorted((u, v)))  # type: ignore[return-value]


def build_graph(edges: Iterable[Edge]) -> nx.Graph:
    """Build weighted undirected graph from edge triples."""

    graph = nx.Graph()
    for u, v, weight in edges:
        if weight <= 0:
            raise ValueError(f"Edge {u}-{v} has non-positive weight: {weight}")
        graph.add_edge(u, v, weight=float(weight))
    if not nx.is_connected(graph):
        raise ValueError("Chinese Postman solver requires connected graph")
    return graph


def graph_base_cost(graph: nx.Graph) -> float:
    """Sum each original undirected edge weight once."""

    return sum(float(data["weight"]) for _, _, data in graph.edges(data=True))


def find_odd_nodes(graph: nx.Graph) -> list[Node]:
    """Find vertices with odd degree."""

    return sorted(node for node, degree in graph.degree() if degree % 2 == 1)


def shortest_path_lookup(
    graph: nx.Graph, odd_nodes: list[Node]
) -> tuple[dict[Pair, float], dict[Pair, Route]]:
    """Compute weighted shortest-path distances and node paths for odd-node pairs."""

    distances: dict[Pair, float] = {}
    paths: dict[Pair, Route] = {}
    for u, v in itertools.combinations(odd_nodes, 2):
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
) -> tuple[nx.MultiGraph, list[Pair], dict[Pair, Route]]:
    """Add duplicate edges along selected shortest paths to create Eulerian multigraph."""

    eulerized = nx.MultiGraph()
    for u, v, data in graph.edges(data=True):
        eulerized.add_edge(u, v, weight=float(data["weight"]), kind="original")

    duplicate_edges: list[Pair] = []
    selected_paths: dict[Pair, Route] = {}
    for pair in pairs:
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


def solve_chinese_postman(graph: nx.Graph, depot: Node = "D") -> PostmanSolution:
    """Solve undirected Chinese Postman Problem for weighted connected graph."""

    if depot not in graph:
        raise ValueError(f"Depot {depot!r} not found in graph")

    base_cost = graph_base_cost(graph)
    odd_nodes = find_odd_nodes(graph)
    pair_distances, pair_paths = shortest_path_lookup(graph, odd_nodes)
    optimal_pairs = solve_minimum_pairing(odd_nodes, pair_distances)
    added_cost = sum(pair_distances[pair] for pair in optimal_pairs)

    eulerized, duplicate_edges, duplicate_paths = duplicate_path_edges(
        graph=graph,
        pairs=optimal_pairs,
        pair_paths=pair_paths,
    )

    final_odds = find_odd_nodes(eulerized)
    if final_odds:
        raise RouteValidationError(f"Eulerized graph still has odd nodes: {final_odds}")
    if not nx.is_eulerian(eulerized):
        raise RouteValidationError("Eulerized graph is not Eulerian")

    circuit_edges = [(u, v) for u, v in nx.eulerian_circuit(eulerized, source=depot)]
    route = [depot]
    route.extend(v for _, v in circuit_edges)
    total_cost = multigraph_route_cost(eulerized, circuit_edges)

    solution = PostmanSolution(
        base_cost=base_cost,
        added_cost=added_cost,
        total_cost=total_cost,
        odd_nodes=odd_nodes,
        optimal_pairs=optimal_pairs,
        duplicate_edges=duplicate_edges,
        duplicate_paths=duplicate_paths,
        route=route,
        circuit_edges=circuit_edges,
    )
    validate_solution(graph, eulerized, solution, depot=depot)
    return solution


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


def edge_weight_labels(graph: nx.Graph) -> dict[Pair, str]:
    """Build edge-label dict for NetworkX drawing."""

    return {
        (u, v): str(
            int(data["weight"])
            if float(data["weight"]).is_integer()
            else data["weight"]
        )
        for u, v, data in graph.edges(data=True)
    }


def draw_base_graph(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_path: Path,
    depot: Node = "D",
) -> None:
    """Save original graph with odd nodes and depot highlighted."""

    fig, ax = plt.subplots(figsize=(16, 9))
    node_colors = []
    for node in graph.nodes:
        if node == depot:
            node_colors.append("gold")
        elif node in solution.odd_nodes:
            node_colors.append("lightcoral")
        else:
            node_colors.append("lightblue")

    nx.draw_networkx_edges(graph, POSITIONS, ax=ax, width=2.0, alpha=0.75)
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        ax=ax,
        node_color=node_colors,
        node_size=1050,
        edgecolors="black",
    )
    nx.draw_networkx_labels(graph, POSITIONS, ax=ax, font_size=14, font_weight="bold")
    nx.draw_networkx_edge_labels(
        graph,
        POSITIONS,
        edge_labels=edge_weight_labels(graph),
        ax=ax,
        font_size=11,
    )
    ax.set_title(
        "Original Street Network\nDepot D in yellow; odd-degree vertices in red",
        fontsize=18,
        fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_pairing_graph(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_path: Path,
) -> None:
    """Save graph with optimal odd-node duplicate paths highlighted."""

    fig, ax = plt.subplots(figsize=(16, 9))
    nx.draw_networkx_edges(graph, POSITIONS, ax=ax, width=1.7, alpha=0.35)

    duplicate_counts = Counter(solution.duplicate_edges)
    for (u, v), count in sorted(duplicate_counts.items()):
        nx.draw_networkx_edges(
            graph,
            POSITIONS,
            edgelist=[(u, v)],
            ax=ax,
            width=3.5 + count,
            edge_color="red",
            arrows=False,
        )

    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        ax=ax,
        node_color="lightblue",
        node_size=1050,
        edgecolors="black",
    )
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        nodelist=solution.odd_nodes,
        ax=ax,
        node_color="lightcoral",
        node_size=1150,
        edgecolors="black",
    )
    nx.draw_networkx_labels(graph, POSITIONS, ax=ax, font_size=14, font_weight="bold")
    nx.draw_networkx_edge_labels(
        graph,
        POSITIONS,
        edge_labels=edge_weight_labels(graph),
        ax=ax,
        font_size=11,
    )
    pair_text = "; ".join(
        f"{u}-{v}: {'->'.join(solution.duplicate_paths[(u, v)])}"
        for u, v in solution.optimal_pairs
    )
    ax.set_title(
        f"Optimal Odd-Node Pairing and Duplicate Paths\n{pair_text}",
        fontsize=17,
        fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_eulerized_graph(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_path: Path,
) -> None:
    """Save original graph plus duplicated edges as red curved overlays."""

    fig, ax = plt.subplots(figsize=(16, 9))
    nx.draw_networkx_edges(graph, POSITIONS, ax=ax, width=2.0, alpha=0.55)
    nx.draw_networkx_edge_labels(
        graph,
        POSITIONS,
        edge_labels=edge_weight_labels(graph),
        ax=ax,
        font_size=11,
    )

    duplicate_counts = Counter(solution.duplicate_edges)
    for index, ((u, v), count) in enumerate(sorted(duplicate_counts.items())):
        radius = 0.18 + 0.07 * index
        nx.draw_networkx_edges(
            graph,
            POSITIONS,
            edgelist=[(u, v)],
            ax=ax,
            width=2.8 + count,
            edge_color="red",
            connectionstyle=f"arc3,rad={radius}",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
        )

    node_colors = ["gold" if node == "D" else "lightgreen" for node in graph.nodes]
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        ax=ax,
        node_color=node_colors,
        node_size=1050,
        edgecolors="black",
    )
    nx.draw_networkx_labels(graph, POSITIONS, ax=ax, font_size=14, font_weight="bold")
    ax.set_title(
        "Eulerized Graph\nAll degrees even; repeated streets shown as red curved arrows",
        fontsize=18,
        fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_final_route(
    graph: nx.Graph,
    solution: PostmanSolution,
    out_path: Path,
    depot: Node = "D",
) -> None:
    """Save final closed tour with directed arrows and step labels."""

    fig, ax = plt.subplots(figsize=(16, 9))
    nx.draw_networkx_edges(graph, POSITIONS, ax=ax, width=1.4, alpha=0.25)
    nx.draw_networkx_nodes(
        graph,
        POSITIONS,
        ax=ax,
        node_color=["gold" if n == depot else "lightblue" for n in graph.nodes],
        node_size=1050,
        edgecolors="black",
    )
    nx.draw_networkx_labels(graph, POSITIONS, ax=ax, font_size=14, font_weight="bold")
    nx.draw_networkx_edge_labels(
        graph,
        POSITIONS,
        edge_labels=edge_weight_labels(graph),
        ax=ax,
        font_size=10,
    )

    parallel_seen: Counter[Pair] = Counter()
    for step, (u, v) in enumerate(solution.circuit_edges, start=1):
        key = canonical_edge(u, v)
        parallel_seen[key] += 1
        radius = 0.08 * ((parallel_seen[key] + 1) // 2)
        if parallel_seen[key] % 2 == 0:
            radius *= -1
        color = "red" if parallel_seen[key] > 1 else "navy"
        nx.draw_networkx_edges(
            graph,
            POSITIONS,
            edgelist=[(u, v)],
            ax=ax,
            width=2.2,
            edge_color=color,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            connectionstyle=f"arc3,rad={radius}",
            min_source_margin=22,
            min_target_margin=22,
        )
        x1, y1 = POSITIONS[u]
        x2, y2 = POSITIONS[v]
        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2 + radius * 0.7
        ax.text(
            label_x,
            label_y,
            str(step),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.22",
                "fc": "white",
                "ec": color,
                "alpha": 0.85,
            },
        )

    short_route = " -> ".join(solution.route)
    if len(short_route) > 150:
        short_route = short_route[:147] + "..."
    ax.set_title(
        f"Final Closed Tour: {int(solution.total_cost)} minutes\n"
        "Blue = first traversal, red = repeated traversal",
        fontsize=17,
        fontweight="bold",
    )
    ax.text(0.5, -0.07, short_route, transform=ax.transAxes, ha="center", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_cost_breakdown(solution: PostmanSolution, out_path: Path) -> None:
    """Save base/additional/total cost bar chart."""

    fig, ax = plt.subplots(figsize=(16, 9))
    labels = ["Base: all streets once", "Added: repeated streets", "Total route"]
    values = [solution.base_cost, solution.added_cost, solution.total_cost]
    bars = ax.bar(labels, values)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{int(value)} min",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    ax.set_ylabel("Minutes", fontsize=13)
    ax.set_title("Chinese Postman Cost Breakdown", fontsize=18, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_visualizations(
    graph: nx.Graph, solution: PostmanSolution, out_dir: Path
) -> list[Path]:
    """Save all plot artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "01_original_graph.png",
        out_dir / "02_odd_pair_shortest_paths.png",
        out_dir / "03_eulerized_graph.png",
        out_dir / "04_final_route.png",
        out_dir / "05_cost_breakdown.png",
    ]
    draw_base_graph(graph, solution, paths[0])
    draw_pairing_graph(graph, solution, paths[1])
    draw_eulerized_graph(graph, solution, paths[2])
    draw_final_route(graph, solution, paths[3])
    draw_cost_breakdown(solution, paths[4])
    return paths


def print_graph_audit(graph: nx.Graph, solution: PostmanSolution) -> None:
    """Print solver summary and validation facts."""

    print("\n=== Graph audit ===")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Original streets: {graph.number_of_edges()}")
    print("Degrees:")
    for node, degree in sorted(graph.degree()):
        parity = "odd" if degree % 2 else "even"
        print(f"  {node}: {degree} ({parity})")

    print("\n=== Chinese Postman solution ===")
    print(f"Odd nodes: {solution.odd_nodes}")
    print(f"Optimal odd-node pairs: {solution.optimal_pairs}")
    print("Duplicate paths:")
    for pair in solution.optimal_pairs:
        path = solution.duplicate_paths[pair]
        print(f"  {pair[0]}-{pair[1]}: {' -> '.join(path)}")
    print(f"Base cost: {int(solution.base_cost)} minutes")
    print(f"Added repeat cost: {int(solution.added_cost)} minutes")
    print(f"Total cost: {int(solution.total_cost)} minutes")
    print(f"Final route: {' -> '.join(solution.route)}")

    print("\n=== Fab-routing analogy ===")
    width = max(len(left) for left, _ in FAB_ANALOGY_ROWS)
    for left, right in FAB_ANALOGY_ROWS:
        print(f"  {left:<{width}} -> {right}")


def parse_args() -> argparse.Namespace:
    """Parse command-line args."""

    parser = argparse.ArgumentParser(
        description="Solve and visualize the Green Valley garbage-truck Chinese Postman route.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("garbage_postman_outputs"),
        help="Directory where PNG visualizations will be saved.",
    )
    parser.add_argument(
        "--depot",
        default="D",
        help="Start/end node for closed route.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    graph = build_graph(STREET_EDGES)
    solution = solve_chinese_postman(graph, depot=args.depot)
    saved = save_visualizations(graph, solution, args.out_dir)
    print_graph_audit(graph, solution)
    print("\n=== Saved visualizations ===")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
