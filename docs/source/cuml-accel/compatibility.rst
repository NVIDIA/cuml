.. _compatibility:

.. _limitations:

Accelerated Estimator Support
=============================

``cuml.accel`` accelerates the estimators listed below. Unsupported estimators
and operations continue to use their original CPU implementations.
Click an estimator name to display its full list of limitations.

.. _general-behavior:

General Behavior
----------------

**Compatibility**
   The accelerator is tested with ``scikit-learn`` versions 1.6 through 1.9,
   ``umap-learn`` versions 0.5.7 through 0.5.12, and ``hdbscan`` versions 0.8.39
   through 0.8.44. When ``cuml.accel`` detects a version outside these ranges,
   it issues a runtime warning and continues. The untested version will likely
   still work, but it has not been validated and may have subtle compatibility
   issues. Some estimators use scikit-learn's experimental `Array API`_ support
   and require ``scikit-learn`` 1.8 or newer for GPU acceleration.

**CPU fallback**
   Estimators not listed below use their original CPU implementations. Listed
   estimators may also fall back for particular methods, hyperparameters, input
   types, or dependency versions. These fallbacks should be transparent and
   preserve the original workflow. :doc:`logging-and-profiling` explains how to
   identify which operations ran on the GPU and why others fell back to the CPU.

**Results**
   GPU and CPU implementations should provide comparable model quality, but
   they are not guaranteed to be numerically identical. Parallel floating-point
   operations may run in a different order, and some GPU algorithms use
   implementations designed for highly parallel hardware. ``cuml.accel`` also
   does not always generate the full set of fitted attributes that scikit-learn
   does. Missing attributes are typically inspection aids, such as ``n_iters_``,
   rather than values required for inference. Compare appropriate model-quality
   metrics, such as ``model.score`` or accuracy, instead of comparing fitted
   coefficients directly.

**Performance**
   Performance depends on the estimator and workload. Larger datasets generally
   benefit most. On small inputs, initialization and data-transfer costs can
   dominate, limiting the benefit of acceleration or even making execution
   slower. Some GPU operations use runtime-compiled kernels, so their first
   invocation may include compilation overhead. Warm up the operation before
   timing it, and compare end-to-end runtime with and without ``cuml.accel``.
   See :doc:`benchmarks` for representative results.

**Errors and warnings**
   Error and warning messages may differ from scikit-learn, and some errors may
   include C++ stack traces. Exceptions, failures, or measurably worse model
   quality are bugs and should be reported by `opening an issue <open an issue_>`_.


.. _sklearn-limitations:

scikit-learn
------------

sklearn.cluster
~~~~~~~~~~~~~~~

The algorithms used in cuML differ from those in scikit-learn. As such, you
shouldn't expect the fitted attributes (e.g. ``labels_``) to numerically match
an estimator fitted without ``cuml.accel``.

To compare results between estimators, we recommend comparing scores like
``sklearn.metrics.adjusted_rand_score`` or
``sklearn.metrics.adjusted_mutual_info_score``. For low-dimensional data, you
can also visually inspect the resulting cluster assignments.

.. dropdown:: ``KMeans``
   :name: kmeans

   ``KMeans`` will fall back to CPU in the following cases:

   - If a callable ``init`` is provided.
   - If ``X`` is sparse.


.. dropdown:: ``SpectralClustering``
   :name: spectralclustering

   ``SpectralClustering`` will fall back to CPU in the following cases:

   - If ``assign_labels`` is not ``"kmeans"``.
   - If ``affinity`` is not ``"nearest_neighbors"`` or ``"precomputed"``.
     Note that the default value of ``affinity`` in scikit-learn is ``"rbf"``,
     which is not GPU-accelerated.
   - If ``X`` is sparse.

   The following fitted attributes are currently not computed:

   - ``affinity_matrix_``


.. dropdown:: ``DBSCAN``
   :name: dbscan

   ``DBSCAN`` will fall back to CPU in the following cases:

   - If ``algorithm`` isn't ``"auto"`` or ``"brute"``.
   - If ``metric`` isn't one of the supported metrics (``"l2"``, ``"euclidean"``, ``"cosine"``, ``"precomputed"``).
   - If ``X`` is sparse.

   Additional notes:

   - ONNX export via ``skl2onnx`` is not supported for this estimator.


sklearn.covariance
~~~~~~~~~~~~~~~~~~

.. dropdown:: ``EmpiricalCovariance``
   :name: empiricalcovariance

   ``EmpiricalCovariance`` has no known estimator-specific ``cuml.accel``
   limitations.

.. dropdown:: ``LedoitWolf``
   :name: ledoitwolf

   ``LedoitWolf`` has no known estimator-specific ``cuml.accel`` limitations.


sklearn.decomposition
~~~~~~~~~~~~~~~~~~~~~

The ``sklearn.decomposition`` implementations used by ``cuml.accel`` use
different SVD solvers than the ones in scikit-learn, which may result in
numeric differences in the ``components_`` and ``explained_variance_`` values.
These differences should be small for most algorithms, but may be larger for
randomized or less-numerically-stable solvers like ``"randomized"`` or
``"covariance_eigh"``.

.. dropdown:: ``PCA``
   :name: pca

   ``PCA`` will fall back to CPU in the following cases:

   - If ``n_components=0``.
   - If ``n_components="mle"``.

   Additional notes:

   - Parameters for the ``"randomized"`` solver, such as ``random_state``,
     ``n_oversamples``, and ``power_iteration_normalizer``, are ignored.


.. dropdown:: ``IncrementalPCA``
   :name: incrementalpca

   ``IncrementalPCA`` has no known estimator-specific ``cuml.accel`` limitations.

   Additional notes:

   - ``partial_fit`` does not support sparse input. This matches scikit-learn;
     use ``fit`` for sparse input or provide dense batches to ``partial_fit``.


.. dropdown:: ``TruncatedSVD``
   :name: truncatedsvd

   ``TruncatedSVD`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.


   Additional notes:

   - Parameters for the ``"randomized"`` solver, such as ``random_state``,
     ``n_oversamples``, and ``power_iteration_normalizer``, are ignored.


sklearn.ensemble
~~~~~~~~~~~~~~~~

The random forest implementation used by ``cuml.accel`` differs algorithmically
from the one in scikit-learn. As such, you
shouldn't expect the fitted attributes (e.g. ``estimators_``) to numerically match
an estimator fitted without ``cuml.accel``.

To compare results between estimators, we recommend comparing scores like
``sklearn.metrics.root_mean_squared_error`` (for regression) or
``sklearn.metrics.log_loss`` (for classification).

.. dropdown:: ``RandomForestClassifier``
   :name: randomforestclassifier

   ``RandomForestClassifier`` will fall back to CPU in the following cases:

   - If ``criterion`` is ``"log_loss"``.
   - If ``oob_score`` is a callable.
   - If ``warm_start=True``.
   - If ``monotonic_cst`` is not ``None``.
   - If ``max_samples`` is an integer.
   - If ``min_weight_fraction_leaf`` is not ``0``.
   - If ``ccp_alpha`` is not ``0``.
   - If ``class_weight`` is not ``None``.
   - If ``sample_weight`` is passed to ``fit`` or ``score``.
   - If ``X`` is sparse.
   - If ``X`` contains missing values (represented as ``NaN``).
   - If ``y`` is a multi-output target.


.. dropdown:: ``RandomForestRegressor``
   :name: randomforestregressor

   ``RandomForestRegressor`` will fall back to CPU in the following cases:

   - If ``criterion`` is ``"absolute_error"`` or ``"friedman_mse"``.
   - If ``oob_score`` is a callable.
   - If ``warm_start=True``.
   - If ``monotonic_cst`` is not ``None``.
   - If ``max_samples`` is an integer.
   - If ``min_weight_fraction_leaf`` is not ``0``.
   - If ``ccp_alpha`` is not ``0``.
   - If ``sample_weight`` is passed to ``fit`` or ``score``.
   - If ``X`` is sparse.
   - If ``X`` contains missing values (represented as ``NaN``).
   - If ``y`` is a multi-output target.


sklearn.kernel_ridge
~~~~~~~~~~~~~~~~~~~~

.. dropdown:: ``KernelRidge``
   :name: kernelridge

   ``KernelRidge`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - If ``kernel`` is not a string.

   ``KernelRidge`` results should be almost identical to those of scikit-learn
   when running with ``cuml.accel`` enabled. In particular, the fitted
   ``dual_coef_`` should be close enough that they may be compared via
   ``np.allclose``.


sklearn.linear_model
~~~~~~~~~~~~~~~~~~~~

The linear model solvers used by ``cuml.accel`` differ from those used in
scikit-learn. As such, you shouldn't expect the fitted attributes (e.g.
``coef_``) to numerically match an estimator fitted without ``cuml.accel``. For
some estimators (e.g. ``LinearRegression``) you might get a close match, but
for others there may be larger numeric differences.

To compare results between estimators, we recommend comparing model quality
scores like ``sklearn.metrics.r2_score`` (for regression) or
``sklearn.metrics.accuracy_score`` (for classification).

.. dropdown:: ``LinearRegression``
   :name: linearregression

   ``LinearRegression`` will fall back to CPU in the following cases:

   - If ``positive=True``.

   Additionally, the following fitted attributes are currently not computed:

   - ``rank_``
   - ``singular_``


.. dropdown:: ``LogisticRegression``
   :name: logisticregression

   ``LogisticRegression`` will fall back to CPU in the following cases:

   - If ``warm_start=True``.
   - If ``intercept_scaling`` is not ``1``.
   - If the deprecated ``multi_class`` parameter is used.
   - If a callback is configured with ``set_callbacks``.


.. dropdown:: ``ElasticNet``
   :name: elasticnet

   ``ElasticNet`` will fall back to CPU in the following cases:

   - If ``positive=True``.
   - If ``warm_start=True``.
   - If ``precompute`` is not ``False``.

   Additionally, the following fitted attributes are currently not computed:

   - ``dual_gap_``


.. dropdown:: ``Ridge``
   :name: ridge

   ``Ridge`` will fall back to CPU in the following cases:

   - If ``positive=True`` or ``solver="lbfgs"``.


.. dropdown:: ``Lasso``
   :name: lasso

   ``Lasso`` will fall back to CPU in the following cases:

   - If ``positive=True``.
   - If ``warm_start=True``.
   - If ``precompute`` is not ``False``.

   Additionally, the following fitted attributes are currently not computed:

   - ``dual_gap_``


sklearn.manifold
~~~~~~~~~~~~~~~~

.. dropdown:: ``TSNE``
   :name: tsne

   ``TSNE`` will fall back to CPU in the following cases:

   - If ``n_components`` is not ``2``.
   - If ``init`` is an array.
   - If ``init="pca"`` and ``X`` is sparse.
   - If ``metric`` isn't one of the supported metrics (``"l2"``, ``"euclidean"``,
     ``"sqeuclidean"``, ``"cityblock"``, ``"l1"``, ``"manhattan"``,
     ``"minkowski"``, ``"chebyshev"``, ``"cosine"``, ``"correlation"``).

   Additional notes:

   - Even with a ``random_state``, the TSNE implementation used by ``cuml.accel``
     isn't completely deterministic.
   - ONNX export via ``skl2onnx`` is not supported for this estimator.

   While the exact numerical output for TSNE may differ from that obtained without
   ``cuml.accel``, we expect the result *quality* to be approximately as
   good in most cases. Beyond comparing the visual representation, you may find
   comparing the trustworthiness score (computed via
   ``sklearn.manifold.trustworthiness``) or the ``kl_divergence_`` fitted
   attribute useful.


.. dropdown:: ``SpectralEmbedding``
   :name: spectralembedding

   ``SpectralEmbedding`` will fall back to CPU in the following cases:

   - If ``affinity`` is not ``"nearest_neighbors"`` or ``"precomputed"``.
   - If ``X`` is sparse.
   - If ``X`` has only one feature.


   The following fitted attributes are currently not computed:

   - ``affinity_matrix_``

   Additional notes:

   - ONNX export via ``skl2onnx`` is not supported for this estimator.


sklearn.neighbors
~~~~~~~~~~~~~~~~~

.. dropdown:: ``NearestNeighbors``
   :name: nearestneighbors

   ``NearestNeighbors`` will fall back to CPU in the following cases:

   - If ``metric`` is not one of the supported metrics (``"l2"``,
     ``"euclidean"``, ``"l1"``, ``"cityblock"``, ``"manhattan"``, ``"taxicab"``,
     ``"canberra"``, ``"minkowski"``, ``"lp"``, ``"chebyshev"``, ``"linf"``,
     ``"jensenshannon"``, ``"cosine"``, ``"correlation"``, ``"inner_product"``,
     ``"sqeuclidean"``, ``"haversine"``).

   Additional notes:

   - The ``algorithm`` parameter is ignored. cuML always uses the GPU-accelerated
     ``"brute"`` implementation.

   - cuML does not implement ``radius_neighbors``, so this method always falls
     back to the CPU.

   - ONNX export via ``skl2onnx`` is not supported for this estimator.


.. dropdown:: ``KNeighborsClassifier``
   :name: kneighborsclassifier

   ``KNeighborsClassifier`` will fall back to CPU in the following cases:

   - If ``metric`` is not one of the supported metrics (``"l2"``,
     ``"euclidean"``, ``"l1"``, ``"cityblock"``, ``"manhattan"``, ``"taxicab"``,
     ``"canberra"``, ``"minkowski"``, ``"lp"``, ``"chebyshev"``, ``"linf"``,
     ``"jensenshannon"``, ``"cosine"``, ``"correlation"``, ``"inner_product"``,
     ``"sqeuclidean"``, ``"haversine"``).

   Additional notes:

   - The ``algorithm`` parameter is ignored. cuML always uses the GPU-accelerated
     ``"brute"`` implementation.


.. dropdown:: ``KNeighborsRegressor``
   :name: kneighborsregressor

   ``KNeighborsRegressor`` will fall back to CPU in the following cases:

   - If ``metric`` is not one of the supported metrics (``"l2"``,
     ``"euclidean"``, ``"l1"``, ``"cityblock"``, ``"manhattan"``, ``"taxicab"``,
     ``"canberra"``, ``"minkowski"``, ``"lp"``, ``"chebyshev"``, ``"linf"``,
     ``"jensenshannon"``, ``"cosine"``, ``"correlation"``, ``"inner_product"``,
     ``"sqeuclidean"``, ``"haversine"``).

   Additional notes:

   - The ``algorithm`` parameter is ignored. cuML always uses the GPU-accelerated
     ``"brute"`` implementation.


.. dropdown:: ``KernelDensity``
   :name: kerneldensity

   ``KernelDensity`` will fall back to CPU in the following cases:

   - If ``metric`` is not one of the supported metrics (``"cityblock"``,
     ``"cosine"``, ``"euclidean"``, ``"l1"``, ``"l2"``, ``"manhattan"``,
     ``"sqeuclidean"``, ``"canberra"``, ``"chebyshev"``, ``"minkowski"``,
     ``"hellinger"``, ``"correlation"``, ``"jensenshannon"``, ``"hamming"``,
     ``"kldivergence"``, ``"russellrao"``, ``"nan_euclidean"``).

   Additional notes:

   - The ``algorithm``, ``atol``, ``rtol``, ``breadth_first``, and ``leaf_size``
     parameters are ignored. The GPU accelerated pairwise brute-force
     implementation in cuML will always be used.


sklearn.preprocessing
~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: ``StandardScaler``
   :name: standardscaler

   ``StandardScaler`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - When run with ``scikit-learn<1.8``.
   - If a callback is configured with ``set_callbacks``.


.. dropdown:: ``MinMaxScaler``
   :name: minmaxscaler

   ``MinMaxScaler`` will fall back to CPU in the following cases:

   - When run with ``scikit-learn<1.8``.


.. dropdown:: ``MaxAbsScaler``
   :name: maxabsscaler

   ``MaxAbsScaler`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - When run with ``scikit-learn<1.8``.


.. dropdown:: ``PolynomialFeatures``
   :name: polynomialfeatures

   ``PolynomialFeatures`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - If ``order`` is ``"F"``.
   - When run with ``scikit-learn<1.8``.


.. dropdown:: ``LabelEncoder``
   :name: labelencoder

   ``LabelEncoder`` has no known estimator-specific ``cuml.accel`` limitations.


.. dropdown:: ``LabelBinarizer``
   :name: labelbinarizer

   ``LabelBinarizer`` has no known estimator-specific ``cuml.accel`` limitations.


.. dropdown:: ``OneHotEncoder``
   :name: onehotencoder

   ``OneHotEncoder`` will fall back to CPU in the following cases:

   - If any columns in ``X`` have ``bytes`` values (``str`` values work fine).
   - If ``dtype`` is not a float or bool dtype.
   - If ``drop`` is ``"if_binary"``
   - If ``handle_unknown`` is ``"warn"`` or ``"infrequent_if_exist"``.
   - If ``min_frequency`` is not ``None``.
   - If ``max_categories`` is not ``None``.
   - If ``feature_name_combiner`` is a callable.

   Additional notes:

   - cuML's encoder treats ``None`` and ``NaN`` values as identical, while
     scikit-learn's encoder treats these as different categories.


.. dropdown:: ``TargetEncoder``
   :name: targetencoder

   ``TargetEncoder`` will fall back to CPU in the following cases:

   - If ``categories`` is not ``"auto"``.
   - If ``y`` is a multiclass target.
   - If ``random_state`` is a ``numpy.random.RandomState`` object (integer seeds
     work fine).

   Additional notes:

   - cuML and scikit-learn use different cross-validation fold assignment strategies
     during ``fit_transform``. Both are valid target encoding implementations, but
     samples are assigned to different folds, resulting in different leave-fold-out
     encodings for training data. The ``transform`` method on test data produces
     equivalent results since it uses global statistics computed from all training
     samples.


sklearn.svm
~~~~~~~~~~~

The SVM implementations used by ``cuml.accel`` differ from those in scikit-learn.
As such,
you shouldn't expect the fitted attributes (e.g. ``coef_`` or
``support_vectors_``) to numerically match an estimator fitted without
``cuml.accel``.

To compare results between estimators, we recommend comparing model quality
scores like ``sklearn.metrics.r2_score`` (for regression) or
``sklearn.metrics.accuracy_score`` (for classification).

.. dropdown:: ``SVC``
   :name: svc

   ``SVC`` will fall back to CPU in the following cases:

   - If ``kernel="precomputed"`` or is a callable.
   - If ``y`` is multiclass.
   - If ``probability=True``. The ``probability`` parameter is deprecated in
     ``scikit-learn>=1.9``, as well as in ``cuml>=26.06``. We recommend wrapping
     ``SVC`` with ``sklearn.calibration.CalibratedClassifierCV``, for example
     ``CalibratedClassifierCV(SVC(), ensemble=False)``. This approach is supported
     across scikit-learn versions and does not require CPU fallback.


.. dropdown:: ``SVR``
   :name: svr

   ``SVR`` will fall back to CPU in the following cases:

   - If ``kernel="precomputed"`` or is a callable.


.. dropdown:: ``LinearSVC``
   :name: linearsvc

   ``LinearSVC`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - If ``intercept_scaling`` is not ``1``.
   - If ``multi_class`` is not ``"ovr"``.

   Additional notes:

   - Use of sample weights may not produce exactly equivalent results when
     compared to replicating data according to weights.


.. dropdown:: ``LinearSVR``
   :name: linearsvr

   ``LinearSVR`` will fall back to CPU in the following cases:

   - If ``X`` is sparse.
   - If ``intercept_scaling`` is not ``1``.

   Additional notes:

   - Use of sample weights may not produce exactly equivalent results when
     compared to replicating data according to weights.


UMAP
----

.. dropdown:: ``UMAP``
   :name: umap-limitations

   ``UMAP`` will fall back to CPU in the following cases:

   - If ``init`` is not ``"random"`` or ``"spectral"``.
   - If ``metric`` is not one of the supported metrics (``"l1"``, ``"cityblock"``,
     ``"taxicab"``, ``"manhattan"``, ``"euclidean"``, ``"l2"``, ``"sqeuclidean"``,
     ``"canberra"``, ``"minkowski"``, ``"chebyshev"``, ``"linf"``, ``"cosine"``,
     ``"correlation"``, ``"hellinger"``, ``"hamming"``, ``"jaccard"``).
   - If ``target_metric`` is not one of the supported metrics (``"categorical"``,
     ``"l2"``, ``"euclidean"``).
   - If ``unique=True``.
   - If ``densmap=True``.
   - If ``ensure_all_finite`` is not ``True``.

   Additional notes:

   - Reproducibility with the use of a seed (the ``random_state`` parameter) comes
     at the relative expense of performance.

   - Parallelism during the optimization stage introduces numerical imprecision,
     which can lead to differences between CPU and GPU results.

   - ONNX export via ``skl2onnx`` is not supported for this estimator.

   - We have observed compatibility issues with UMAP when using ``numba>=0.62.0``.
     For best stability, we recommend using ``numba<0.62.0`` when accelerating
     UMAP with ``cuml.accel``.

   While the exact numerical output for UMAP may differ from that obtained without
   ``cuml.accel``, we expect the result *quality* to be approximately as
   good in most cases. Beyond comparing the visual representation, you may find
   comparing the trustworthiness score (computed via
   ``sklearn.manifold.trustworthiness``) useful.


HDBSCAN
-------

.. dropdown:: ``sklearn.cluster.HDBSCAN``
   :name: sklearn-hdbscan-limitations

   ``sklearn.cluster.HDBSCAN`` will fall back to CPU in the following cases:

   - If ``metric`` is not ``"l2"`` or ``"euclidean"``.
   - If ``metric_params`` is not empty.
   - If ``store_centers`` is not ``None``.
   - If the effective scikit-learn ``min_samples`` value is outside the range
     of 2 through 1024. In particular, ``min_samples=1`` falls back to CPU.
   - If the input is sparse, precomputed, or contains non-finite values.

   The ``algorithm``, ``leaf_size``, ``n_jobs``, and ``copy`` parameters control
   CPU implementation details and do not change GPU execution for supported
   dense inputs. The public ``dbscan_clustering`` method runs on CPU using the
   linkage tree computed during the GPU fit; it does not refit the estimator.

   Additional notes:

   - ``max_cluster_size=None`` is translated to cuML's unlimited value.
   - cuML uses float32 computation and a parallel MST, so linkage distances,
     probabilities, cluster labels, and even cluster assignments may differ
     numerically from scikit-learn's CPU implementation.
   - cuML's ``HDBSCAN`` implementation uses a parallel MST, which means
     the results are not deterministic when there are duplicates in the mutual
     reachability graph.
   - ONNX export via ``skl2onnx`` is not supported for this estimator.

.. dropdown:: ``hdbscan.HDBSCAN``
   :name: hdbscan-limitations

   ``hdbscan.HDBSCAN`` will fall back to CPU in the following cases:

   - If ``metric`` is not ``"l2"`` or ``"euclidean"``.
   - If a ``memory`` location is configured.
   - If ``match_reference_implementation=True``.
   - If ``branch_detection_data=True``.

   Additionally, the following fitted attributes are currently not computed:

   - ``exemplars_``
   - ``outlier_scores_``
   - ``relative_validity_``

   Additional notes:

   - cuML uses float32 computation and a parallel MST, so linkage distances,
     probabilities, cluster labels, and even cluster assignments may differ
     numerically from the contrib CPU implementation.
   - cuML's ``HDBSCAN`` implementation uses a parallel MST, which means
     the results are not deterministic when there are duplicates in the mutual
     reachability graph.
   - ONNX export via ``skl2onnx`` is not supported for this estimator.


.. _open an issue: https://github.com/NVIDIA/cuml/issues
.. _Array API: https://scikit-learn.org/stable/modules/array_api.html
