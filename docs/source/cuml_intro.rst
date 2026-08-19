Introduction
============

cuML's design is built around three core principles that work together to make
GPU-accelerated machine learning both powerful and accessible. These principles
guide every aspect of the library's development and determine how you interact
with cuML in practice. Understanding these principles will help you take full
advantage of cuML's capabilities and write more effective code.

1. :ref:`Where possible, match familiar APIs <where-possible-match-the-scikit-learn-api>`
2. :ref:`Be flexible on inputs and predictable on outputs <accept-flexible-input-types-return-predictable-output-types>`
3. :ref:`Deliver top GPU-accelerated performance <be-fast>`

These principles work together to create a seamless experience that combines the familiarity of scikit-learn and other popular ML packages with the power of GPU acceleration. Let's explore how each principle shapes cuML's design and functionality.

Familiar APIs for Seamless Integration
--------------------------------------

.. _where-possible-match-the-scikit-learn-api:

cuML estimators look and feel just like `scikit-learn estimators
<https://scikit-learn.org/stable/developers/develop.html>`_. This means:

* **Familiar workflow**: Initialize with parameters, fit with data, predict/transform for inference
* **Drop-in replacement**: Most sklearn code works with minimal changes
* **Consistent naming**: Same method names and parameter conventions
* **Compatible interfaces**: Works with sklearn's cross-validation, pipelines, and preprocessing

.. code-block:: python

   from cuml import LinearRegression
   from cuml.datasets import make_regression
   from cuml.model_selection import train_test_split

   # Generate sample data directly in GPU memory
   X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)

   # Split data into train and test datasets
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

   # Fit model
   model = LinearRegression()
   model.fit(X_train, y_train)

   # Make predictions
   y_pred = model.predict(X_test)
   print(f"R² score: {model.score(X_test, y_test):.3f}")

You can find many more complete examples in the `Introductory Notebook
<estimator_intro.ipynb>`__ and in the cuML API documentation.

.. note::

   While cuML's APIs closely mirror scikit-learn, some differences exist due to
   GPU-specific implementations and optimizations.

   For **zero code change acceleration** with 100% API compatibility, use
   `cuml.accel <cuml-accel/index.rst>`_.

Flexible Inputs, Predictable Outputs
------------------------------------

.. _accept-flexible-input-types-return-predictable-output-types:

cuML estimators can accept NumPy arrays, cuDF dataframes, cuPy arrays,
2D PyTorch tensors, Python lists and tuples, and really any kind of
standards-based Python array input you can throw at them. This relies on
the ``__array__`` and ``__cuda_array_interface__`` standards, widely used
throughout the PyData community, and on cuML's input validation helpers.
When you provide a Python list or tuple, cuML validates and converts it
just like other host-backed array inputs.

By default, outputs will mirror the data type you provided. So, if you
fit a model with a NumPy array, the ``model.coef_`` property
containing fitted coefficients will also be a NumPy array. If you fit
a model using cuDF's GPU-based DataFrame and Series objects, the
model's output properties will be cuDF objects. You can always
override this behavior and select a default datatype with
:func:`cuml.set_global_output_type`.

The `RAPIDS Configurable Input and Output Types
<https://medium.com/@dantegd/e719d72c135b>`_ blog post goes into much
more detail explaining this approach.

GPU-Accelerated Performance
---------------------------

.. _be-fast:

cuML transforms slow, CPU-bound machine learning workflows into fast, interactive
experiences. What takes minutes or hours on the CPU often completes in seconds
on the GPU, enabling real-time experimentation and rapid iteration. On average,
you can expect **10-50x performance improvements** on demanding workflows. cuML
delivers this performance through:

* **Highly-optimized CUDA primitives** and algorithms
* **GPU-accelerated implementations** designed for modern hardware
* **Efficient memory management** and data movement

Performance gains vary by algorithm and dataset size:

* **4x faster** for medium-sized linear regression
* **1000x+ faster** for large-scale t-SNE dimensionality reduction
* **Scaling benefits** increase with larger datasets

.. note::
   Modern GPUs have 5000+ cores. To maximize performance, ensure you're providing
   enough data to keep the GPU busy. Expect larger performance gains as dataset
   size grows.

The :doc:`cuml.benchmark <api/cuml.benchmark>` module provides an easy interface
to benchmark your own hardware.


What's Next
===========

Here are some suggestions on what to explore next:

1. **Try the examples**: Walk through the `Introductory Notebook
   <estimator_intro.ipynb>`__ for hands-on learning
2. **Explore the API**: Browse the `API Reference <api>`_ for specific algorithms
3. **Check out notebooks**: Try examples in the `notebooks <https://github.com/NVIDIA/cuml/tree/main/notebooks>`_ directory
4. **Learn advanced topics**: Read the `cuML blogs <cuml_blogs.rst>`_ for deeper insights
5. **Get help**: Visit our `GitHub Issues <https://github.com/NVIDIA/cuml/issues>`_
   or join the `CUDA-X Data Science Community <https://developer.nvidia.com/topics/ai/data-science/cuda-x-for-data-science#join-the-community>`_
