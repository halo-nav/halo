# HALo

HALo is a Python package for Model Predictive Path Integral (MPPI) information driven navigation for scalar fields. It implments a Fisher information based information driven navigation system.

It uses a Localizability metric that is constructed from a rank-recovered Fisher information matrix in a scalar field. This is due the rank deficiency associated with taking the gradient in a scalar field with respect to the state of the robot. The rank-recovered Fisher information matrix is used to construct a localizability metric that is used in the MPPI cost function to drive the robot towards regions of high localizability.

## Project structure

```text
halo/
├── README.md
├── pyproject.toml
├── main.py                     # Interactive MPPI navigation example
├── halo/
│   ├── __init__.py
│   ├── mppi.py                 # Model Predictive Path Integral controller
│   ├── particle_filter.py      # Particle-filter localization
│   └── reduced_rank_gp.py      # Reduced-rank Gaussian Process model
├── halo-data/                  # Git submodule containing map data/tools
│   ├── README.md
│   ├── pyproject.toml
│   ├── map_generator.py        # Synthetic map generator
│   └── data/
│       ├── mag.csv
│       ├── synthetic_scalar.csv
│       └── synthetic_scalar_single.csv
└── .gitmodules
```

## Setup

The project requires Python 3.12 or newer. From the repository root, install
the dependencies with:

```bash
uv sync
```

If the `halo-data` submodule was not checked out, initialize it with:

```bash
git submodule update --init --recursive
```

## Running the navigation example

`main.py` runs an interactive MPPI navigation example using the magnetic map at
`halo-data/data/mag.csv`:

```bash
uv run python main.py
```

The equivalent command after activating the project environment is:

```bash
python main.py
```

`main.py` does not currently define command-line flags. Its parameters (map
path, GP basis size, MPPI horizon and sample count, goal, initial state, and
plot bounds) are configured in `main.py`. The example opens a live Matplotlib
plot and continues until it is interrupted with `Ctrl+C`.

## Generating synthetic map data

The data submodule has its own dependencies. From `halo-data/`, install them
and run the generator:

```bash
cd halo-data
uv sync
uv run python map_generator.py
```

The generator writes a CSV map, a modified CSV map, and a JSON file containing
start/end poses to the current directory. Use `--plot` to also save a PNG
visualization.

### `map_generator.py` flags

| Flag | Default | Description |
| --- | --- | --- |
| `--output-dir PATH` | `.` | Directory in which generated files are written |
| `--name NAME` | `synthetic_scalar_single` | Base name for the generated CSV and PNG files |
| `--seed INTEGER` | `42` | Seed for reproducible map and pose generation |
| `--pose-count INTEGER` | `50` | Number of start/end poses to generate; must not be negative |
| `--plot` | disabled | Also save a PNG visualization of the generated field |
| `-h`, `--help` | — | Show all available options |

For example:

```bash
uv run python map_generator.py \
  --output-dir ./generated \
  --name experiment \
  --seed 7 \
  --pose-count 100 \
  --plot
```

This creates `experiment.csv`, `experiment_modified.csv`,
`synthetic_scalar_field_start_end_positions.json`, and, with `--plot`,
`experiment.png` in `./generated`.

## Python API

The controllers and GP model can also be imported directly:

```python
from halo import MPPI, ParticleFilter, ReducedRankGP
```