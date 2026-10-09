cuml.benchmark
==============

The built-in harness runs YAML-defined estimator workloads and writes JSON
results files. Start with the
:doc:`CLI benchmarking guide </developer_guide/benchmarking>` for commands,
suite/profile selection, manifest examples, timing semantics, and resume rules.

Programmatic execution
----------------------

The public runner works directly in scripts and notebook cells, using the same
isolated provider execution as the CLI.

.. code-block:: python

    from cuml import benchmark

    results = benchmark.run(
        "my-suite.yaml", providers=["cuml", "scikit-learn"]
    )

``run()`` returns a ``BenchmarkResults`` dataclass. Its ``artifacts`` field holds
JSON dictionaries keyed by provider, for example ``results.artifacts["cuml"]``.
The defaults are the
``estimators`` suite, ``standard`` profile, and ``cuml`` provider. Customize
workloads and execution counts through YAML.

Pass a directory path to ``output`` to save results and checkpoints, for example
``output="./results"``. Use the same directory with ``resume=True`` to continue a
run. Omitting ``output`` uses temporary storage.

.. autofunction:: cuml.benchmark.run

.. autoclass:: cuml.benchmark.BenchmarkResults

.. autoclass:: cuml.benchmark.BenchmarkRunError
