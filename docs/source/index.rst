NVIDIA cuML Documentation
=========================

NVIDIA cuML is a suite of fast, GPU-accelerated machine learning algorithms
designed for data science and analytical tasks. Our API mirrors scikit-learn,
providing practitioners with the familiar fit-predict-transform paradigm
without requiring GPU programming expertise. With `cuml.accel`, cuML can also
automatically accelerate existing code with zero code changes.

cuML delivers on average **10-50x faster performance** than CPU-based
alternatives for realistic workloads and supports **50+ algorithms** across all
major machine learning categories, including clustering, regression,
classification, dimensionality reduction, and time series analysis. With
comprehensive **multi-GPU and multi-node support** via Dask, cuML scales from
single workstations to large clusters.

Especially if your scikit-learn, umap-learn, or hdbscan workflows take many
minutes to complete, you will likely benefit from using cuML. The equivalent
cuML estimators often run in seconds.


Quick Start
===========

.. code-block:: python

   from cuml.datasets import make_blobs
   from cuml.cluster import DBSCAN

   # Create sample data
   X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

   # Fit clustering model
   dbscan = DBSCAN(eps=1.0, min_samples=5)
   dbscan.fit(X)
   print(dbscan.labels_)

Key Features
============

* **GPU Acceleration**: 10-50x faster than CPU-based alternatives
* **Scikit-learn Compatible**: Drop-in replacement for most sklearn algorithms
* **Multi-GPU Support**: Scale across multiple GPUs and nodes with Dask
* **Comprehensive Coverage**: 50+ algorithms across all major ML categories
* **Flexible Input**: Works with NumPy, cuDF, cuPy, and PyTorch tensors
* **Production Ready**: Battle-tested in enterprise environments

Installation
============

cuML is available through conda and pip. For detailed installation instructions,
visit the `RAPIDS Release Selector <https://docs.rapids.ai/install#selector>`_.

.. note::
   cuML is only supported on Linux operating systems and WSL 2. See
   `the RAPIDS install page <https://docs.rapids.ai/install/#system-req>`_
   for details on system and hardware requirements.

CUDA-X Data Science
===================

NVIDIA cuML is an open-source CUDA-X Data Science library for GPU-accelerated
machine learning. It integrates with libraries in the broader RAPIDS ecosystem,
including cuDF for data manipulation and cuGraph for graph analytics.

Community & Support
===================

* :doc:`User Guide <user_guide>` - Comprehensive usage documentation
* :doc:`API Reference <api/index>` - Complete API documentation
* `GitHub Issues <https://github.com/NVIDIA/cuml/issues>`_ - Report bugs and request features
* `CUDA-X Data Science Community <https://developer.nvidia.com/topics/ai/data-science/cuda-x-for-data-science#join-the-community>`_ - Join our community

.. toctree::
   :hidden:

   cuml_intro.rst
   user_guide.rst
   Zero Code Change Acceleration <cuml-accel/index.rst>
   api/index
   cuml_blogs.rst
