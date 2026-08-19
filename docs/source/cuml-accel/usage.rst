Getting Started with cuml.accel
===============================

``cuml.accel`` intercepts supported scikit-learn, UMAP, and HDBSCAN estimator
operations and dispatches them to GPU implementations. When an operation
cannot run on the GPU, it transparently uses the original CPU implementation
instead. A fallback may depend on the estimator, method, parameter values,
input data, or installed library version. See :doc:`compatibility`
for the accelerated estimator inventory and exact fallback conditions.

CPU fallback preserves compatibility, but frequent transitions between CPU
and GPU execution may reduce the overall speedup. The
:doc:`logging and profiling tools <logging-and-profiling>` show which operations
ran on the GPU, which fell back to the CPU, and why.

.. _activation-methods:

Activation Methods
------------------

Enable ``cuml.accel`` before importing scikit-learn, UMAP, or HDBSCAN. When
running a script, use the ``cuml.accel`` command-line interface:

.. code-block:: console

   python -m cuml.accel script.py

In Jupyter or IPython, load the extension before other imports:

.. code-block:: ipython

   %load_ext cuml.accel

.. _cuml-accel-env-var:

For third-party applications whose code you do not control, set
``CUML_ACCEL_ENABLED`` to ``1`` or ``true`` (case insensitive):

.. code-block:: console

   CUML_ACCEL_ENABLED=1 python script.py

This loads the accelerator for every Python program launched with the variable
set and may add startup overhead. If cuML is not installed correctly, the
environment variable is silently ignored and execution remains on the CPU, so
the CLI or notebook extension is usually easier to validate. See
:doc:`examples/third-party-apps` for more.

You can also install the accelerator programmatically before importing the
libraries it will accelerate:

.. code-block:: python

   import cuml

   cuml.accel.install()

.. _what-results-to-expect:

For result-equivalence and performance guidance, see
:ref:`general-behavior`.

Memory Management
-----------------

When the platform supports it and RMM has not already been configured,
``cuml.accel`` enables `managed memory`_. Managed memory can use host memory in
addition to GPU memory and migrates data as needed, reducing the risk of GPU
out-of-memory errors. It does not prevent exhaustion of combined host and
device memory, and heavy oversubscription can slow execution. Managed memory
is not enabled on WSL 2 or when RMM was configured before ``cuml.accel``.
If managed-memory oversubscription causes unexpectedly slow execution, use the
CLI's ``--disable-uvm`` flag to disable it and compare performance.

For workloads that always run on NVIDIA GPUs, using cuML directly may provide
more control over GPU-specific parameters and device-memory usage. Direct cuML
can also expose functionality beyond the compatible APIs provided through
``cuml.accel``. Start with ``cuml.accel`` when retaining an existing codebase
matters; consider direct cuML when GPU-specific tuning and control are more
important.

.. _managed memory: https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
