..
   This page is a Jinja template. The ``deps.*`` version constraints are
   filled in from dependencies.yaml at build time by
   docs/source/sphinxext/dependency_versions.py. To change a version, edit
   dependencies.yaml; to add a package, add it here and to ``PACKAGES`` there.

Supported Versions
==================

Please see https://docs.nvidia.com/datascience/install/ for ecosystem-wide version support.

We aim to meet the `SPEC 0 guidelines <https://scientific-python.org/specs/spec-0000/>`_ for minimal supported versions.

Required Runtime Dependencies
-----------------------------

The following dependencies are required for the cuML library:

* **NumPy**: {{ deps.numpy }}
* **scikit-learn**: {{ deps.scikit_learn }}
* **scipy**: {{ deps.scipy }}
* **cupy**: {{ deps.cupy }}
* **treelite**: {{ deps.treelite }}

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

* **xgboost**: {{ deps.xgboost }} (for gradient boosting algorithms)
* **hdbscan**: {{ deps.hdbscan }} (for hierarchical density-based clustering)
* **umap-learn**: {{ deps.umap_learn }} (for dimensionality reduction)
* **pynndescent**: {{ deps.pynndescent }} (for approximate nearest neighbor search)
* **numba-cuda-mlir**: {{ deps.numba_cuda_mlir }} (for user defined metrics)

CUDA-X Dependencies
-------------------

Some cuML dependencies within the CUDA-X Data Science ecosystem are pinned to
the same version. For example, cuML 25.08 is compatible with and only with cuDF
25.08, cuVS 25.08, rmm 25.08, etc. These constraints are expressed as part of
the package dependencies and aren't something a user should typically need to
worry about manually ensuring.
