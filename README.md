# Cross-Snapshot Attention Graph Neural Network

Novel cross-snapshot attention method that leverages the unique features of propagations originating from specific nodes over time. The novelty lies in the element-wise attention weight calculations across consecutive snapshots, linking changes in propagation states to local network regions.

## Installation

The project uses poetry for package management but also provides a docker setup. Installation via poetry is recommended for development, while docker can be used to quickly execute experiments.
Recommended python version is 3.10.

<details>
<summary>Poetry</summary>

**- Install dependencies.**

`poetry install`

**NOTE:** PyTorch Geometric is not part of the poetry project and needs to be installed separately for your machine. See [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

**- Start virtual environment that can be used to execute commands.**

`poetry shell`

**- Run command XXX in the poetry shell.**

`python3 XXX`

</details>

<details>
<summary>Docker</summary>
In contrast to poetry, PyTorch Geometric is automatically included with docker.

**- Create a new image.**

`docker build -t csa .`

**- Run command XXX in new container.**

`docker compose run csa python3 XXX `

</details>

## Source Code

The propagation generation framework is implemented via a dedicated `Propagation` class that maintains propagation snapshots as NetworkX graphs. Datasets are generated based on distinct class configurations, designated as `Labels`. The generic `Dataset` interface creates datasets, writing a `Sample` composed of a `Propagation` and a `Label` to disk. Models are constructed using PyTorch and PyTorch Geometric, requiring graph transformation into a specific format. Thus, a `TransformedDatasetFactory` resides over the dataset interface, producing an InMemoryDataset in PyG format, derived from the original propagation graphs. It should be noted that dataset transformations are regenerated only if the transformed files are absent or if the dataset is created with the `recreate` flag. Manual updates to the dataset files or failure of previous transformations will not trigger automatic recreation.
The folder structure should be self-explanatory.

## Experiments

This repository contains a large set of experiments.

- `classification`: reports accuracy of the models on the default synthesized datasets
- `complexity`: investigates model behavior when increasing behavior graph complexity
- `curriculum`: investigates model behavior when training models using a set of curriculums
- `datasets`: produces statistics and visualizations around the generated synthetic datasets
- `density`: investigates model behavior when increasing behavior graph density
- `explainability`: draws attention weights for some specific samples from the synthesized datasets
- `masking`: investigates model behavior when using different masking strategies on the propagation samples
- `perturbations`: investigates model behavior when using different perturbation strategies on the propagation samples and behavior graphs
- `pretext`: investigates different pretext tasks for the self-supervised prompting model that recovers snapshots
- `runtime`: reports runtime of the different models on the default synthesized datasets
- `transportability`: investigates model behavior when transporting models between different domains (e.g., training on a specific set of behavior graphs and testing on a different one)

Each experiment is implemented as its own python module. For instance, `classification` representing model evaluation on the base datasets can be run like this (inside poetry or docker):

`python3 -m experiments.classification.runner`

All modules implement the same command line arguments:

- `--runs X`: specifies number of experiment runs
- `--analyze`: creates the main analysis (i.e., charts) for the experiment
- `--stats`: performs additional statistic tests (e.g., regressions) if implemented for this experiment
- `--no-execute`: skips execution of the experiment, can be used in conjunction with `analyze` if only the charts should be rebuilt
- `--parallelize`: parallelizes experimental runs using the maximal number of workers available on the system (based on Ray)
