# NVIDIA cuML: GPU-Accelerated Machine Learning

NVIDIA cuML is an open-source CUDA-X Data Science library for
GPU-accelerated machine learning. It provides two ways to run machine learning
workloads on NVIDIA GPUs:

- The `cuml` Python API provides GPU-native estimators with familiar
  scikit-learn-style APIs and direct control over machine learning workflows.
- `cuml.accel` accelerates existing scikit-learn, UMAP, and HDBSCAN code without
  changing the Python code that uses those libraries.

On representative benchmarks, cuML can accelerate scikit-learn workflows by
up to 50x. Performance depends on the algorithm, dataset, and hardware. See the
[cuML benchmarks](https://docs.nvidia.com/cuml/26.08/cuml-accel/benchmarks/) for
results and methodology.

## Use the GPU-native `cuml` API

The `cuml` Python API follows the familiar scikit-learn fit-predict-transform pattern while keeping data and computation on the GPU. The following example generates sample data and computes DBSCAN clusters on the GPU:

```python
from cuml.datasets import make_blobs
from cuml.cluster import DBSCAN

# Create sample data
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Fit clustering model
dbscan = DBSCAN(eps=1.0, min_samples=5)
dbscan.fit(X)
print(dbscan.labels_)
```

`cuml` supports clustering, dimensionality reduction, regression,
classification, preprocessing, model selection, time series, model
explanation, and nearest-neighbor workflows. Browse the [API
reference](https://docs.nvidia.com/cuml/26.08/api/) for the current list of
estimators and functions.

## Accelerate existing code with `cuml.accel`

Run an existing Python script through the [`cuml.accel` module](https://docs.nvidia.com/cuml/26.08/cuml-accel/):

```console
python -m cuml.accel script.py
```

Or load the extension in a Jupyter notebook before importing scikit-learn,
UMAP, or HDBSCAN:

```python
%load_ext cuml.accel
```

Supported operations run on the GPU. When an estimator or configuration cannot
be accelerated, `cuml.accel` uses the original CPU implementation so the rest
of the workflow can continue. See the [`cuml.accel` compatibility
documentation](https://docs.nvidia.com/cuml/26.08/cuml-accel/compatibility/) for
current coverage and fallback conditions. Use the [logging and profiling
tools](https://docs.nvidia.com/cuml/26.08/cuml-accel/logging-and-profiling/) to check
which operations ran on the GPU.

## Scale beyond one GPU

The `cuml.dask` API provides distributed implementations of selected algorithms
for multi-GPU and multi-node execution with [Dask](https://www.dask.org). See
the [multi-GPU guide](https://docs.nvidia.com/cuml/26.08/dask_multigpu_guide/) for
cluster setup, supported algorithms, and examples.

## Installation

Use the [installation selector](https://docs.rapids.ai/install#selector) to
generate a command for installing nightly or release cuML packages with conda,
pip, or Docker.

Additional resources:

- [NVIDIA cuML documentation](https://docs.nvidia.com/cuml/)
- [NVIDIA cuML product page](https://developer.nvidia.com/topics/ai/data-science/cuda-x-data-science-libraries/cuml)
- [Walkthrough notebooks](https://github.com/NVIDIA/cuml/tree/main/notebooks)
- [CUDA-X Data Science libraries](https://developer.nvidia.com/topics/ai/data-science/cuda-x-for-data-science)

## Build and install from source

See the [build guide](BUILD.md).

## Scikit-learn compatibility

cuML is compatible with scikit-learn version 1.6 or higher.

## Model serialization and security

cuML models can be serialized with `pickle` or `joblib` and loaded later for
inference. cuML uses cloudpickle so that models trained with `cuml.accel` can be
loaded and used with scikit-learn.

**Only unpickle or deserialize from trusted sources.** The `pickle` module (and
by extension `joblib`) is not secure: malicious payloads can execute arbitrary
code during deserialization and compromise your system. **Do not unpickle or
load data from untrusted or tampered sources.** This applies to `pickle.load()`,
`pickle.loads()`, `joblib.load()`, and any file-based model loading. For
details and patterns, see the [Model Serialization and
Persistence](docs/source/pickling_cuml_models.ipynb) notebook and the [Python
pickle security documentation](https://docs.python.org/3/library/pickle.html).

## Contributing and support

See the [contributing guide](CONTRIBUTING.md) to contribute to cuML. Report bugs
and request features through [GitHub issues](https://github.com/NVIDIA/cuml/issues).
Join the broader community through the [CUDA-X Data Science libraries
page](https://developer.nvidia.com/topics/ai/data-science/cuda-x-for-data-science#join-the-community).

## Citation

For additional details on the technologies behind cuML and the broader Python
machine learning landscape, see [_Machine Learning in Python: Main developments
and technology trends in data science, machine learning, and artificial
intelligence_ (2020)](https://arxiv.org/abs/2002.04803) by Sebastian Raschka,
Joshua Patterson, and Corey Nolet.

Please consider citing this work when using cuML in a project:

```bibtex
@article{raschka2020machine,
  title={Machine Learning in Python: Main developments and technology trends in data science, machine learning, and artificial intelligence},
  author={Raschka, Sebastian and Patterson, Joshua and Nolet, Corey},
  journal={arXiv preprint arXiv:2002.04803},
  year={2020}
}
```
