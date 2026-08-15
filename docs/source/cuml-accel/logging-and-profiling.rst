Logging and Profiling
=====================

``cuml.accel`` provides logging and profiling support to help you understand
which operations are being accelerated on GPU and which are falling back to CPU
execution. This can be particularly useful for debugging performance issues or
understanding why certain operations might not be accelerated.

Logging
^^^^^^^

The logging system provides different levels of detail:

* **WARN level**: Shows only warnings and errors. The default level.
* **INFO level**: Shows information on where dispatched methods ran (GPU or CPU) and why.
* **DEBUG level**: Shows more detailed information about GPU initialization,
  parameter synchronization, attribute synchronization, and all method calls.

To enable logging, you can set the logging level in several ways:

Command Line Interface (CLI)
----------------------------

When running scripts with ``cuml.accel``, you can use the ``-v`` or ``--verbose`` flag:

.. code-block:: console

   # Show warnings only (default)
   python -m cuml.accel myscript.py

   # Show info level logs
   python -m cuml.accel -v myscript.py

   # Show debug level logs (most verbose)
   python -m cuml.accel -vv myscript.py

Programmatic Installation
-------------------------

When using the programmatic installation method, you can set the log level directly:

.. code-block:: python

   import cuml

   # Install with debug logging
   cuml.accel.install(log_level="debug")

   # Install with info logging
   cuml.accel.install(log_level="info")

   # Install with warning logging (default)
   cuml.accel.install(log_level="warn")

Jupyter/IPython
---------------

The log level may be set in Jupyter notebooks or IPython through the
``cuml.accel.log_level`` line magic after loading the ``cuml.accel`` extension:

.. code-block:: ipython

   # Load the cuml.accel extension, enabling cuml.accel
   %load_ext cuml.accel

   # Set the log level to info
   %cuml.accel.log_level info

Environment Variable
~~~~~~~~~~~~~~~~~~~~

The log level may also be configured by setting the ``CUML_ACCEL_LOG_LEVEL``
environment variable to ``warn``, ``info`` or ``debug`` (case insensitive).
This will be used if no explicit level is passed through either
``cuml.accel.install`` or the CLI.

Example
-------

.. code-block:: python

   from sklearn.linear_model import Ridge
   from sklearn.datasets import make_regression

   X, y = make_regression()

   # Fit and predict on GPU
   ridge = Ridge(alpha=1.0)
   ridge.fit(X, y)
   ridge.predict(X)

   # Retry, using a hyperparameter that isn't supported on GPU
   ridge = Ridge(positive=True)
   ridge.fit(X, y)
   ridge.predict(X)


Executing this with ``python -m cuml.accel -v script.py`` will show the following output:

.. code-block:: console

   [cuml.accel] Accelerator installed.
   [cuml.accel] `Ridge.fit` ran on GPU
   [cuml.accel] `Ridge.predict` ran on GPU
   [cuml.accel] `Ridge.fit` falling back to CPU: `positive=True` is not supported
   [cuml.accel] `Ridge.fit` ran on CPU
   [cuml.accel] `Ridge.predict` ran on CPU

This logging information can help you:

* Identify which parts of your pipeline are being accelerated
* Understand why certain operations fall back to CPU
* Debug performance issues by seeing where GPU acceleration fails
* Optimize your code by understanding synchronization patterns

Profiling
^^^^^^^^^

In addition to logging, ``cuml.accel`` contains two profilers to help users better
understand what parts of their code ``cuml.accel`` was able to accelerate.

Function Profiler
-----------------

The function profiler gathers statistics about potentially accelerated function
and method calls. It can show:

- Which method calls ``cuml.accel`` had the potential to accelerate (if any).
  Note that only methods ``cuml.accel`` can currently accelerate are included
  in this table (even if a CPU fallback was required). Methods that are fully
  unimplemented won't be present.
- Which methods were accelerated on GPU, and their total runtime.
- Which methods required a CPU fallback, their total runtime, and why a
  fallback was needed

.. warning::

   The function profiler does not currently track GPU calls made in
   subprocesses. This may happen when using meta-estimators like
   ``RandomizedSearchCV``. For now we recommend avoiding setting ``n_jobs > 1``
   when using the profilers.

It can be enabled in a few different ways:

**Command Line Interface (CLI)**

If running using the CLI, you may add the ``--profile`` flag to profile your
whole script.

.. code-block:: console

    python -m cuml.accel --profile script.py

**Jupyter Notebook**

If running in IPython or Jupyter, you may use the ``cuml.accel.profile`` cell
magic to profile code running in a single cell.

.. code-block:: ipython

   %%cuml.accel.profile

   # All code in this cell will be profiled
   ...

**Programmatic Usage**

Alternatively, the ``cuml.accel.profile`` contextmanager may be used to
programmatically profile a section of code.

.. code-block:: python

    with cuml.accel.profile():
        # All code within this context will be profiled
        ...

In all cases, once the profiler's context ends, a report will be generated.

For example, running the following script:

.. code-block:: python

   from sklearn.linear_model import Ridge
   from sklearn.datasets import make_regression

   X, y = make_regression(n_samples=100)

   # Fit and predict on GPU
   ridge = Ridge(alpha=1.0)
   ridge.fit(X, y)
   ridge.predict(X)

   # Retry, using a hyperparameter that isn't supported on GPU
   ridge = Ridge(positive=True)
   ridge.fit(X, y)
   ridge.predict(X)

as ``python -m cuml.accel --profile script.py`` will output the following report

.. code-block:: text

    cuml.accel profile
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
    ┃ Function      ┃ GPU calls ┃ GPU time ┃ CPU calls ┃ CPU time ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
    │ Ridge.fit     │         1 │    171ms │         1 │    4.7ms │
    │ Ridge.predict │         1 │    1.2ms │         1 │   89.8µs │
    ├───────────────┼───────────┼──────────┼───────────┼──────────┤
    │ Total         │         2 │  172.2ms │         2 │    4.8ms │
    └───────────────┴───────────┴──────────┴───────────┴──────────┘
    Not all operations ran on the GPU. The following functions required CPU
    fallback for the following reasons:

    * Ridge.fit
      - `positive=True` is not supported
    * Ridge.predict
      - Estimator not fit on GPU

From this you can see that:

- The only methods ``cuml.accel`` had the potential to accelerate were
  ``Ridge.fit`` and ``Ridge.predict``.
- Each method was called 2 times - once on GPU and once on CPU
- The reason the CPU callback was required was that ``positive=True`` wasn't
  supported.

Line Profiler
-------------

The line profiler collects per-line statistics on your script. It can show:

- Which lines took the most cumulative time.
- Which lines (if any) were able to benefit from acceleration.
- The percentage of each line's runtime that was spent on GPU through ``cuml.accel``.

.. warning::

   The line profiler can add non-negligible overhead. It can be useful to
   gather information on what parts of your code were accelerated, but you
   shouldn't compare runtimes when run with the line profiler enabled to
   other runs.

   Additionally, it does not currently track GPU calls made in subprocesses.
   This may happen when using meta-estimators like ``RandomizedSearchCV``. For
   now we recommend avoiding setting ``n_jobs > 1`` when using the profilers.

**Command Line Interface (CLI)**

If running using the CLI, you may add the ``--line-profile`` flag to run the
line profiler on your whole script.

.. code-block:: console

    python -m cuml.accel --line-profile script.py

**Jupyter Notebook**

If running in IPython or Jupyter, you may use the ``cuml.accel.line_profile``
cell magic to run the line profiler on code in a single cell.

.. code-block:: ipython

   %%cuml.accel.line_profile

   # All code in this cell will be profiled
   ...

In all cases, once the profiler's context ends, a report will be generated.

For example, running the following script:

.. code-block:: python

   from sklearn.linear_model import Ridge
   from sklearn.datasets import make_regression

   X, y = make_regression(n_samples=100)

   # Fit and predict on GPU
   ridge = Ridge(alpha=1.0)
   ridge.fit(X, y)
   ridge.predict(X)

   # Retry, using a hyperparameter that isn't supported on GPU
   ridge = Ridge(positive=True)
   ridge.fit(X, y)
   ridge.predict(X)

as ``python -m cuml.accel --line-profile script.py`` will output the following report

.. code-block:: text

    cuml.accel line profile
    ┏━━━━┳━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃  # ┃ N ┃    Time ┃ GPU % ┃ Source                                       ┃
    ┡━━━━╇━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │  1 │ 1 │       - │     - │ from sklearn.linear_model import Ridge       │
    │  2 │ 1 │       - │     - │ from sklearn.datasets import make_regression │
    │  3 │   │         │       │                                              │
    │  4 │ 1 │   1.5ms │     - │ X, y = make_regression(n_samples=100)        │
    │  5 │   │         │       │                                              │
    │  6 │   │         │       │ # Fit and predict on GPU                     │
    │  7 │ 1 │       - │     - │ ridge = Ridge(alpha=1.0)                     │
    │  8 │ 1 │ 158.4ms │  99.0 │ ridge.fit(X, y)                              │
    │  9 │ 1 │   1.4ms │  97.0 │ ridge.predict(X)                             │
    │ 10 │   │         │       │                                              │
    │ 11 │   │         │       │ # Retry, using an unsupported hyperparameter │
    │ 12 │ 1 │       - │     - │ ridge = Ridge(positive=True)                 │
    │ 13 │ 1 │   6.3ms │   0.0 │ ridge.fit(X, y)                              │
    │ 14 │ 1 │ 153.4µs │   0.0 │ ridge.predict(X)                             │
    └────┴───┴─────────┴───────┴──────────────────────────────────────────────┘
    Ran in 168.3ms, 94.5% on GPU


From this you can see that:

- The first calls to ``Ridge.fit`` and ``Ridge.predict`` (lines 8 and 9) ran on
  GPU, while the latter calls to these same methods (lines 13 and 14) fell back
  to CPU. No other lines had the opportunity for GPU acceleration.

- The script ran in 168.3 ms, 94.5% of which was spent on GPU method calls.
  High percentages here *may* indicate a good fit for ``cuml.accel``, as a
  majority of the total time is spent in accelerated methods. However, even a
  low percentage here may still be worthwhile if the total time taken by the
  script is reduced compared to running without ``cuml.accel``.

- The time taken by the GPU accelerated calls is *much higher* than the time
  taken by the equivalent CPU calls. This is because we're running on very
  small data here, where the overhead of CPU <-> GPU transfer dominates the
  runtime. So while we had a high percentage of utilization (usually good!),
  the runtimes indicate that this particular script may be better run without
  ``cuml.accel``. For this estimator, acceleration does become more beneficial
  once run on larger problems (try increasing to ``n_samples=10_000``).
