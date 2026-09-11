Developer Guide
===============

This guide collects the information needed to contribute to, develop, test,
document, and benchmark cuML. Start with :doc:`contributing`, then use the
implementation-specific guidance below.

* :doc:`Contributing <contributing>` covers how to propose changes, prepare pull
  requests, run repository checks, and work with continuous integration.
* :doc:`Python development <python/development>` covers style, testing,
  validation, memory management, and Python documentation.

  * :doc:`Python estimator development <python/estimators>` describes the
    ``cuml.Base`` estimator contract and provides implementation patterns.

* :doc:`C++ and CUDA development <cpp/index>` covers source layout,
  resources, testing, and Doxygen documentation.

  * :doc:`Internal C++ API reference <cpp/api/index>` exposes primarily
    internal libcuml interfaces for developers. Prefer the supported
    :doc:`Python API <../api/index>` for applications.

* :doc:`Benchmarking <benchmarking>` explains the benchmark CLI and manifests.

.. toctree::
   :hidden:
   :maxdepth: 3

   contributing
   python/development
   python/estimators
   cpp/index
   benchmarking
