# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuML - GPU Machine Learning Algorithms</div>

cuML is a suite of libraries that implement machine learning algorithms and mathematical primitives functions that share compatible APIs with other [RAPIDS](https://rapids.ai/) projects.

cuML enables data scientists, researchers, and software engineers to run
traditional tabular ML tasks on GPUs without going into the details of CUDA
programming. In most cases, cuML's Python API matches the API from
[scikit-learn](https://scikit-learn.org).

For large datasets, these GPU-based implementations can complete 10-50x faster
than their CPU equivalents. For details on performance, see the [cuML Benchmarks
Notebook](https://github.com/NVIDIA/cuml/tree/main/notebooks/tools).

As an example, the following Python snippet loads input and computes DBSCAN clusters, all on GPU, using cuDF:
```python
import cudf
from cuml.cluster import DBSCAN

# Create and populate a GPU DataFrame
gdf_float = cudf.DataFrame()
gdf_float['0'] = [1.0, 2.0, 5.0]
gdf_float['1'] = [4.0, 2.0, 1.0]
gdf_float['2'] = [4.0, 2.0, 1.0]

# Setup and fit clusters
dbscan_float = DBSCAN(eps=1.0, min_samples=1)
dbscan_float.fit(gdf_float)

print(dbscan_float.labels_)
```

Output:
```
0    0
1    1
2    2
dtype: int32
```

cuML also features multi-GPU and multi-node-multi-GPU operation, using [Dask](https://www.dask.org), for a
growing list of algorithms. The following Python snippet reads input from a CSV file and performs
a NearestNeighbors query across a cluster of Dask workers, using multiple GPUs on a single node:


Initialize a `LocalCUDACluster` configured with [UCXX](https://github.com/rapidsai/ucxx) for fast transport of CUDA arrays
```python
# Initialize UCX for high-speed transport of CUDA arrays
from dask_cuda import LocalCUDACluster

# Create a Dask single-node CUDA cluster w/ one worker per device
cluster = LocalCUDACluster(protocol="ucx",
                           enable_tcp_over_ucx=True,
                           enable_nvlink=True,
                           enable_infiniband=False)
```

Load data and perform `k-Nearest Neighbors` search. `cuml.dask` estimators also support `Dask.Array` as input:
```python

from dask.distributed import Client
client = Client(cluster)

# Read CSV file in parallel across workers
import dask_cudf
df = dask_cudf.read_csv("/path/to/csv")

# Fit a NearestNeighbors model and query it
from cuml.dask.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors = 10, client=client)
nn.fit(df)
neighbors = nn.kneighbors(df)
```

For additional examples, browse our complete [API
documentation](https://docs.nvidia.com/cuml/), or check out our
example [walkthrough
notebooks](https://github.com/NVIDIA/cuml/tree/main/notebooks). Finally, you
can find complete end-to-end examples in the [notebooks-contrib
repo](https://github.com/rapidsai/notebooks-contrib).


### Supported Algorithms
| Category | Algorithm | Notes |
| --- | --- | --- |
| **Clustering** |  Density-Based Spatial Clustering of Applications with Noise (DBSCAN) | Multi-node multi-GPU via Dask |
|  | Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)  | |
|  | K-Means | Multi-node multi-GPU via Dask |
|  | Single-Linkage Agglomerative Clustering | |
|  | Spectral Clustering | |
| **Dimensionality Reduction** | Principal Components Analysis (PCA) | Multi-node multi-GPU via Dask|
| | Incremental PCA | |
| | Truncated Singular Value Decomposition (tSVD) | Multi-node multi-GPU via Dask |
| | Uniform Manifold Approximation and Projection (UMAP) | Multi-node multi-GPU Inference via Dask |
| | Random Projection | |
| | t-Distributed Stochastic Neighbor Embedding (TSNE) | |
| | Spectral Embedding | |
| **Linear Models for Regression or Classification** | Linear Regression (OLS) | Multi-node multi-GPU via Dask |
| | Linear Regression with Lasso or Ridge Regularization | Multi-node multi-GPU via Dask |
| | ElasticNet Regression | |
| | LARS Regression | (experimental) |
| | Logistic Regression | Multi-node multi-GPU via Dask-GLM [demo](https://github.com/daxiongshu/rapids-demos) |
| | Naive Bayes | Multi-node multi-GPU via Dask |
| | Stochastic Gradient Descent (SGD), Coordinate Descent (CD), and Quasi-Newton (QN) (including L-BFGS and OWL-QN) solvers for linear models  | |
| **Nonlinear Models for Regression or Classification** | Random Forest (RF) Classification | Experimental multi-node multi-GPU via Dask |
| | Random Forest (RF) Regression | Experimental multi-node multi-GPU via Dask |
| | Inference for decision tree-based models | Forest Inference Library (FIL) |
|  | K-Nearest Neighbors (KNN) Classification | Multi-node multi-GPU via Dask+[UCXX](https://github.com/rapidsai/ucxx), uses [Faiss](https://github.com/facebookresearch/faiss) for Nearest Neighbors Query. |
|  | K-Nearest Neighbors (KNN) Regression | Multi-node multi-GPU via Dask+[UCXX](https://github.com/rapidsai/ucxx), uses [Faiss](https://github.com/facebookresearch/faiss) for Nearest Neighbors Query. |
|  | Support Vector Machine Classifier (SVC) | |
|  | Epsilon-Support Vector Regression (SVR) | |
| **Preprocessing** | Standardization, or mean removal and variance scaling / Normalization / Encoding categorical features / Discretization / Imputation of missing values / Polynomial features generation / and coming soon custom transformers and non-linear transformation | Based on Scikit-Learn preprocessing
| **Time Series** | Holt-Winters Exponential Smoothing | |
|  | Auto-regressive Integrated Moving Average (ARIMA) | Supports seasonality (SARIMA) |
| **Model Explanation** | SHAP Kernel Explainer | [Based on SHAP](https://shap.readthedocs.io/en/latest/) |
|  | SHAP Permutation Explainer | [Based on SHAP](https://shap.readthedocs.io/en/latest/) |
| **Other**                                             | K-Nearest Neighbors (KNN) Search                                                                                                          | Multi-node multi-GPU via Dask+[UCXX](https://github.com/rapidsai/ucxx), uses [Faiss](https://github.com/facebookresearch/faiss) for Nearest Neighbors Query. |

---

## Installation

See [the RAPIDS Release Selector](https://docs.rapids.ai/install#selector) for
the command line to install either nightly or official release cuML packages
via conda, pip, or Docker.

## Build/Install from Source
See the build [guide](BUILD.md).

## Scikit-learn Compatibility

cuML is compatible with scikit-learn version 1.6 or higher.

## Model serialization and security

cuML models can be serialized with `pickle` or `joblib` and loaded later for inference. cuML uses cloudpickle so that models trained with cuml.accel can be loaded and used with scikit-learn.

**Only unpickle or deserialize from trusted sources.** The `pickle` module (and by extension `joblib`) is not secure: malicious payloads can execute arbitrary code during deserialization and compromise your system. **Do not unpickle or load data from untrusted or tampered sources.** This applies to `pickle.load()` / `pickle.loads()`, `joblib.load()`, and any file-based model loading. For details and patterns, see the [Model Serialization and Persistence](docs/source/pickling_cuml_models.ipynb) notebook and the [Python pickle security documentation](https://docs.python.org/3/library/pickle.html).

## Contributing

Please see our [guide for contributing to cuML](CONTRIBUTING.md).

## References

The RAPIDS team has a number of blogs with deeper technical dives and examples. [You can find them here on Medium.](https://medium.com/rapids-ai/tagged/machine-learning)

For additional details on the technologies behind cuML, as well as a broader overview of the Python Machine Learning landscape, see [_Machine Learning in Python: Main developments and technology trends in data science, machine learning, and artificial intelligence_ (2020)](https://arxiv.org/abs/2002.04803) by Sebastian Raschka, Joshua Patterson, and Corey Nolet.

Please consider citing this when using cuML in a project. You can use the citation BibTeX:

```bibtex
@article{raschka2020machine,
  title={Machine Learning in Python: Main developments and technology trends in data science, machine learning, and artificial intelligence},
  author={Raschka, Sebastian and Patterson, Joshua and Nolet, Corey},
  journal={arXiv preprint arXiv:2002.04803},
  year={2020}
}
```

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="80%"/></p>
