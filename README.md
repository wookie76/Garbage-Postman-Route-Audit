# Garbage Postman Route Audit

Chinese Postman route solver and visualization project. Built from a Medium article example, then refactored into a GitHub-style Python artifact with CLI commands, validation, Rich tables, deterministic matplotlib visuals, route diagnostics, and fab-routing interpretation.

This project solves a **closed route planning problem**: traverse every street at least once, return to depot, and minimize repeated travel. In graph terms, this is the **undirected Chinese Postman Problem**, not Traveling Salesman. TSP visits nodes; Chinese Postman covers edges. Source article uses a garbage truck route example and solves it by repairing odd-degree nodes, duplicating shortest paths, and extracting an Eulerian closed tour. 

## Why this project exists

This started as a graph theory exercise, then became a small manufacturing analytics portfolio project.

Core idea:

```text
street edge      -> process-step transition
travel time      -> cycle time, queue time, cost, or risk
depot            -> lot release / route anchor / finish anchor
odd node         -> imbalance point that prevents one-pass closed route
duplicated edge  -> repeated move, rework, inspection, or metrology revisit
closed route     -> complete audit traversal with minimum repeat burden
```

The same graph audit pattern can support **virtual metrology**, **soft sensor routing**, **wafer route diagnostics**, **process-flow bottleneck analysis**, and **repeat-burden auditing**.

## What was built

Final script:

```text
garbage_postman_unified_v3.py
```

Main features:

```text
NetworkX exact Chinese Postman solver
Pydantic v2 config validation
Typer CLI
Rich tables
Loguru logging
Pathlib paths
tqdm progress guard
matplotlib Agg backend
seaborn icefire-inspired palette
16:9 visual outputs
centrality / bottleneck audit
route dashboard
route architecture diagram
CSV and JSON exports
fab / wafer-route analogy command
```

Second Medium PDF contributed architecture/reporting ideas: layered pipeline, graph metrics, feature fusion framing, clustering/dashboard presentation style. We kept useful architecture and dashboard patterns, but rejected NLP-heavy pieces like embeddings, FAISS, UMAP, and topic models because they do not help this route solver. 

## Results

Expected result:

```text
Base street cost:        70 minutes
Repeated street cost:    12 minutes
Total closed route:      82 minutes
Odd-degree nodes:        A, D, G, I
Optimal repair pairs:    A-I, D-G
Route traversals:        20
Original streets:        16
```

Validated locally:

```text
Validation passed: 70 + 12 = 82
```

The solve run also confirmed graph audit tables, centrality audit, matching table, final route metrics, fab mapping, and saved outputs. 

## Installation

```bash
pip install networkx matplotlib seaborn typer rich pydantic loguru tqdm
```

## How to run

### Validate only

```bash
python garbage_postman_unified_v3.py validate --out-dir garbage_postman_outputs_v3
```

Expected:

```text
Validation passed: 70 + 12 = 82
```

### Solve and generate outputs

```bash
python garbage_postman_unified_v3.py solve --out-dir garbage_postman_outputs_v3
```

### Show fab / wafer-route mapping

```bash
python garbage_postman_unified_v3.py fab-map
```

## Output files

```text
garbage_postman_outputs_v3/
├── 01_original_graph.png
├── 02_odd_pair_shortest_paths.png
├── 03_eulerized_graph.png
├── 04_final_route.png
├── 05_cost_breakdown.png
├── 06_route_step_timeline.png
├── 07_route_dashboard.png
├── 08_route_architecture.png
├── route_steps.csv
├── node_centrality.csv
├── duplicated_edges.csv
└── metrics.json
```

## Visual outputs

### `01_original_graph.png`

Original weighted street network. Depot `D` shown in gold. Odd-degree nodes shown in warm red. Edge labels show minutes.

### `02_odd_pair_shortest_paths.png`

Shows minimum added-cost repair paths:

```text
A-I: A → B → E → I
D-G: D → G
```

### `03_eulerized_graph.png`

Shows duplicated streets as warm curved arcs. After duplication, all node degrees become even, so an Eulerian closed route exists.

### `04_final_route.png`

Technical proof plot. Shows final route with arrows and step bubbles. Dense, but useful for inspection.

### `05_cost_breakdown.png`

Simple executive chart:

```text
70 min base + 12 min repeat = 82 min total
```

### `06_route_step_timeline.png`

Readable route sequence timeline. Better than final route plot for human explanation.

### `07_route_dashboard.png`

Analytics dashboard with:

```text
cost breakdown
node degree audit
top bottleneck nodes
final route diagnostics
repeated streets
```

### `08_route_architecture.png`

Pipeline diagram inspired by knowledge-system architecture:

```text
Street / Process Route Data
↓
Graph Ingestion + Validation
↓
Odd-Degree / Imbalance Audit
↓
Shortest Paths + Matching
↓
Eulerized Route Construction
↓
Closed Tour + Metrics
↓
Fab / Wafer Interpretation
```

## Important implementation choices

### Exact solver for demo

Current route uses an exact graph method because assumptions hold:

```text
undirected graph
positive weights
connected network
fixed street costs
all edges required
```

### Heuristics later for fab reality

For real fab routing, exact mathematical optimality may be wrong target. Manufacturing constraints can make a “perfect” graph answer unrealistic.

Future route mode should be:

```text
heuristics-first
fixed time budget
deterministic seed
validated coverage
clearly marked "best found", not "optimal"
```

Possible heuristic cost:

```text
edge_cost =
    cycle_time
  + queue_time_penalty
  + tool_risk_penalty
  + rework_penalty
  + metrology_sampling_penalty
  + drift_penalty
```

## Development history

### v1

Built unified exact solver from article.

### v2

Added:

```text
Typer
Rich tables
Loguru
Pathlib
Pydantic v2
tqdm
matplotlib Agg
seaborn palette
16:9 visual design
```

### v3

Added:

```text
lazy plotting imports
validate --out-dir support
node centrality / bottleneck audit
route dashboard
route architecture diagram
route timeline
CSV/JSON exports
```

## Quality checks

Current validation covers:

```text
depot exists
graph is connected
edge weights are positive
odd node count is even
Eulerian multigraph produced
route starts at depot
route ends at depot
every original street covered
total cost equals expected 82
```

## Known limitations

```text
Only built-in article graph for now
No external CSV route ingestion yet
No directed Chinese Postman support yet
No heuristic fab-route mode yet
No capacity/tool constraints yet
No PuLP/MILP backend yet
```

Best next branch:

```text
v3.1 = add CSV route input + sample fab_route.csv
```



```text
process routing
factory analytics
semiconductor manufacturing
fab operations
equipment data
route diagnostics
graph analytics
model validation
audit trail
explainable analytics
risk scoring
process monitoring
```

### Resume bullet draft

```text
Built a Python graph-analytics route audit using NetworkX, Typer, Pydantic v2, Rich, Loguru, and matplotlib to solve an undirected Chinese Postman routing problem with validation, visual diagnostics, centrality-based bottleneck analysis, and CSV/JSON reporting.
```

```text
Supports configurable process-route networks with weighted transitions for virtual metrology, soft sensor routing, and wafer-flow audit experiments.
```

