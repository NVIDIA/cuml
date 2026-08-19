User Guide
==========

This section provides practical guidance on how to get started with cuML in your
own projects to run classic ML algorithms blazingly fast on NVIDIA GPUs. You can
simply read through the examples or try them out yourself. If you want to run
code directly, have a look at the notebooks within the `notebooks
<https://github.com/NVIDIA/cuml/tree/main/notebooks>`_ as part of the cuML
GitHub repository instead.

.. note::

   This guide describes how to use cuML's GPU-accelerated estimators and
   functions directly within your own code. See the :doc:`the next section
   <cuml-accel/index>` if you want to accelerate existing scikit-learn, UMAP,
   and HDBSCAN code with zero code changes.

.. toctree::
   :maxdepth: 2

   estimator_intro.ipynb
   pickling_cuml_models.ipynb
   dask_multigpu_guide.ipynb
   advanced.rst
   health_checks.rst
   supported_versions.rst
