# Optimal Routing for Quantum Networks — Caleffi (2017)

Replication of the numerical results from
**M. Caleffi, "Optimal Routing for Quantum Networks", IEEE Access, 2017.**

This project implements the link- and end-to-end entanglement rate model
described in the paper, the optimal-routing algorithm (Algorithm 3) and a
Dijkstra/Bellman-Ford baseline, and reproduces the figures from Section V
of the paper.

---

## Project structure

```
qit-project/
├── generate_figures.py         # Entry point — runs all figures and saves them to ./images/
├── Dockerfile                  # Container image for reproducible runs
├── docker-compose.yml          # Compose service that mounts ./images/ on host
├── requirements.txt            # Full project dependencies (incl. SeQUeNCe)
├── requirements-docker.txt     # Minimal deps for figure generation
├── utils/
│   ├── constants/              # Physical constants (PARAMS)
│   ├── models/
│   │   ├── physical_params.py  # PhysicalParams dataclass
│   │   ├── topology.py         # Topology graph model
│   │   ├── optimal_routing.py  # OptimalRouting (Eq. 8, Algorithm 3)
│   │   └── djikstra.py         # DijkstraRouting baseline
│   └── handlers/
│       └── figures.py          # Figures class — one method per paper figure
├── scripts/                    # Standalone exploration / sanity scripts
├── tests/                      # Unit and integration tests
└── tutorials/                  # Jupyter notebooks walking through the model
```

---

## Figures reproduced

| Method               | Paper figure | What it shows                                                   |
| -------------------- | ------------ | --------------------------------------------------------------- |
| `figure4`            | Fig. 4       | Link entanglement rate ξ vs link length, varying τ_d            |
| `figure5`            | Fig. 5       | E2E rate vs route length on a 2-hop route (repeater position)   |
| `figure5_heatmap`    | Fig. 5       | Same as above as a heatmap over (D, α)                          |
| `figure6`            | Fig. 6       | Minimum coherence time τ vs route length                        |
| `figure6_heatmap`    | Fig. 6       | Same as above as a heatmap over (D, α)                          |
| `figure8`            | Fig. 8       | E2E rate of the four routes from Fig. 7                         |
| `figure9`            | Fig. 9       | Optimal (Algorithm 3) vs Dijkstra/Bellman-Ford routing          |

Each method writes a PNG to `./images/<method_name>.png`.

---

## Running locally (Python)

Requirements: Python 3.11+.

```bash
pip install -r requirements.txt
python generate_figures.py
```

Output PNGs land in `./images/`.

---

## Running with Docker

The Docker image installs only the minimal dependencies needed to produce
the figures (`numpy`, `matplotlib`).

```bash
docker compose up --build
```

The compose service mounts `./images/` from the host, so generated figures
appear directly on your machine.

To clean up:

```bash
docker compose down
```

---

## Running individual figures

From a Python session or script:

```python
from utils.handlers.figures import Figures

f = Figures()
f.figure4(save_path="images/figure4.png", show=False)
f.figure9(save_path="images/figure9.png", show=False)
```

Some methods accept a `lines` argument to plot only a subset of curves
(e.g. `figure5(lines=[0, 1])`).

---

## Tests

```bash
pytest tests/
```

---

## Reference

Marcello Caleffi, *Optimal Routing for Quantum Networks*,
IEEE Access, vol. 5, pp. 22299–22312, 2017.
[doi:10.1109/ACCESS.2017.2763325](https://doi.org/10.1109/ACCESS.2017.2763325)
