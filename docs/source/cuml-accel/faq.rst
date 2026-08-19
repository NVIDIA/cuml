Frequently Asked Questions
==========================

How does ``cuml.accel`` work with ``cudf.pandas``?
--------------------------------------------------

``cuml.accel`` provides zero-code-change acceleration for scikit-learn and
similar packages, while ``cudf.pandas`` does the same for pandas. They can be
used together from the command line:

.. code-block:: console

   python -m cudf.pandas -m cuml.accel script.py

In Jupyter, load both extensions before importing the libraries they
accelerate:

.. code-block:: ipython

   %load_ext cudf.pandas
   %load_ext cuml.accel

Can I load a serialized model without ``cuml.accel``?
-----------------------------------------------------

Models serialized while ``cuml.accel`` is active may be loaded without the
accelerator. They are then restored as their normal scikit-learn, UMAP, or
HDBSCAN counterparts. If loaded while ``cuml.accel`` is active, they are
restored as accelerated proxy models.

Serialization formats are not guaranteed to remain compatible across cuML or
dependency versions. For long-term portability, record the training
environment and validate loading and inference in the target environment.

.. warning::

   Only unpickle or deserialize models from trusted sources. Python's
   ``pickle`` module is not secure; malicious data can execute arbitrary code.
   See the `Python pickle documentation`_ for details.

How do I report a bug?
----------------------

Open a report in the `cuML issue tracker`_. Include a minimal reproducer,
dependency versions, input characteristics, and logs from an ``info`` or
``debug`` run when possible. Please report exceptions, incorrect fallback,
measurably worse model quality, and regressions in end-to-end runtime. Import
time is expected to be longer with ``cuml.accel`` and should be excluded from
runtime comparisons.

For help determining what ran on the GPU and why an operation fell back, see
:doc:`logging-and-profiling`. The estimator inventory is in
:doc:`compatibility`, along with detailed estimator-specific
conditions.

.. _Python pickle documentation: https://docs.python.org/3/library/pickle.html
.. _cuML issue tracker: https://github.com/NVIDIA/cuml/issues/new?template=bug_report.md
