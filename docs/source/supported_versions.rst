Supported Versions
==================

Please see https://docs.nvidia.com/datascience/install/ for ecosystem-wide version support.

We aim to meet the `SPEC 0 guidelines <https://scientific-python.org/specs/spec-0000/>`_ for minimal supported versions.

Required Runtime Dependencies
-----------------------------

The following dependencies are required for the cuML library:

* **NumPy**: >=2,<3.0a0
* **scikit-learn**: >=1.6
* **scipy**: >=1.14.0
* **numba**: >=0.60.0,<0.62.0a0
* **cupy**: cupy-cuda12x>=14.0.1 (CUDA 12), cupy-cuda13x>=14.0.1 (CUDA 13)
* **treelite**: >=4.6.1,<5.0.0

Dask Runtime Dependencies
-------------------------

To use the ``cuml.dask`` module, you'll also need to install the following
additional dependencies:

* **rapids-dask-dependency**
* **dask-cudf**
* **raft-dask**

All with constraints matching the requested ``cuml`` version.

Optional Runtime Dependencies
-----------------------------

The following dependencies are optional and provide additional functionality:

* **xgboost**: >=2.1.0 (for gradient boosting algorithms)
* **hdbscan**: >=0.8.39,<0.8.40 (for hierarchical density-based clustering)
* **umap-learn**: >=0.5.7,<0.5.13 (for dimensionality reduction)
* **pynndescent**: (for approximate nearest neighbor search)

CUDA-X Dependencies
-------------------

Some cuML dependencies within the CUDA-X Data Science ecosystem are pinned to
the same version. For example, cuML 25.08 is compatible with and only with cuDF
25.08, cuVS 25.08, rmm 25.08, etc. These constraints are expressed as part of
the package dependencies and aren't something a user should typically need to
worry about manually ensuring.
