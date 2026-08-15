cuml.accel Overview
===================

``cuml.accel`` runs supported scikit-learn, UMAP, and HDBSCAN workloads on an
NVIDIA GPU without changing the Python code that uses those libraries. It is a
good fit when you want to accelerate an existing workflow, keep the familiar
APIs, or quickly evaluate the benefit of GPU acceleration before using cuML
directly.

Enable ``cuml.accel`` before importing the libraries you want to accelerate.
Your existing code then remains unchanged:

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.linear_model import Ridge

   X, y = make_regression(n_samples=1_000_000, random_state=0)
   model = Ridge().fit(X, y)
   predictions = model.predict(X)

Run a script through the ``cuml.accel`` command-line interface:

.. code-block:: console

   python -m cuml.accel script.py

Or load the extension at the top of a Jupyter notebook, before other imports:

.. code-block:: ipython

   %load_ext cuml.accel

See :ref:`activation-methods` for a complete overview of the command-line,
Jupyter, environment-variable, and programmatic activation options.

``cuml.accel`` transparently falls back to the original CPU implementation
when an estimator or operation cannot be accelerated. Fallback can depend on
parameters, input types, methods, or library versions, so existing workflows
continue to run even when only part of a pipeline is GPU accelerated. Use the
:doc:`logging and profiling tools <logging-and-profiling>` to see exactly
where execution occurs.

We have validated ``cuml.accel`` against the complete scikit-learn example
gallery: every runnable example completes successfully with ``cuml.accel``
enabled. This does not mean that every operation runs on the GPU. Unsupported
operations transparently fall back to the CPU. The result demonstrates broad
compatibility with realistic, complex scikit-learn workflows.

Where to Go Next
----------------

* :doc:`usage` explains activation, fallback, and memory management.
* :doc:`compatibility` lists the estimators that can be accelerated,
  including estimator-specific fallback conditions and behavioral differences.
* :doc:`logging-and-profiling` explains how to identify which operations ran on
  the GPU and why others fell back to the CPU.
* :doc:`benchmarks` provides representative performance results.
* :doc:`examples/index` contains complete examples and notebook workflows.
* :doc:`faq` covers interoperability, serialization, and bug reporting.

.. toctree::
   :hidden:

   self
   usage
   compatibility
   logging-and-profiling
   benchmarks
   examples/index
   faq
