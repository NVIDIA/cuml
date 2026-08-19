# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np

import cuml.internals.nvtx as nvtx
from cuml.common.classification import process_class_weight
from cuml.common.doc_utils import generate_docstring, insert_into_docstring
from cuml.ensemble.randomforest_common import BaseRandomForestModel
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.mixins import ClassifierMixin
from cuml.internals.outputs import ClassLabels, ReflectedAttr, mlfunc
from cuml.internals.validation import check_inputs


class RandomForestClassifier(ClassifierMixin, BaseRandomForestModel):
    """
    Implements a Random Forest classifier model which fits multiple decision
    tree classifiers in an ensemble.

    .. note:: Note that the underlying algorithm for tree node splits differs
      from that used in scikit-learn. By default, the cuML Random Forest uses a
      quantile-based algorithm to determine splits, rather than an exact
      count. You can tune the size of the quantiles with the `n_bins`
      parameter.

    .. note:: You can export cuML Random Forest models and run predictions
      with them on machines without an NVIDIA GPUs. See
      https://docs.rapids.ai/api/cuml/nightly/pickling_cuml_models.html
      for more details.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.ensemble import RandomForestClassifier as cuRFC

        >>> X = cp.random.normal(size=(10,4)).astype(cp.float32)
        >>> y = cp.asarray([0,1]*5, dtype=cp.int32)

        >>> cuml_model = cuRFC(max_features=1.0,
        ...                    n_bins=8,
        ...                    n_estimators=40)
        >>> cuml_model.fit(X,y)
        RandomForestClassifier()
        >>> cuml_predict = cuml_model.predict(X)

        >>> print("Predicted labels : ", cuml_predict)
        Predicted labels :  [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]

    Parameters
    ----------
    n_estimators : int (default = 100)
        Number of trees in the forest. (Default changed to 100 i 0.11)
    split_criterion : str or int (default = ``'gini'``)
        The criterion used to split nodes.\n
         * ``'gini'`` or ``0`` for gini impurity
         * ``'entropy'`` or ``1`` for information gain (entropy)
    bootstrap : boolean (default = True)
        Control bootstrapping.\n
            * If ``True``, each tree in the forest is built on a bootstrapped
              sample with replacement.
            * If ``False``, the whole dataset is used to build each tree.
    max_samples : float (default = 1.0)
        Ratio of dataset rows used while fitting each tree.
    max_depth : int or None (default = None)
        Maximum tree depth. Use ``None`` for unlimited depth (trees grow
        until all leaves are pure). Must be a positive integer or ``None``.

        .. rapids-pre-commit-hooks: disable-next-line
        .. versionchanged:: 26.08
          The default of `max_depth` changed from `16` to `None`.
    max_leaves : int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited,
        If ``-1``.
    max_features : {'sqrt', 'log2', None}, int or float (default = 'sqrt')
        The number of features to consider per node split:

        * If an int then ``max_features`` is the absolute count of features to be used.
        * If a float then ``max_features`` is used as a fraction.
        * If ``'sqrt'`` then ``max_features=1/sqrt(n_features)``.
        * If ``'log2'`` then ``max_features=log2(n_features)/n_features``.
        * If ``None`` then ``max_features=n_features``

        .. versionchanged:: 24.06
          The default of `max_features` changed from `"auto"` to `"sqrt"`.

    n_bins : int (default = 128)
        Maximum number of bins used by the split algorithm per feature.
        For large problems, particularly those with highly-skewed input data,
        increasing the number of bins may improve accuracy.
    n_streams : int (default = 4)
        Number of parallel streams used for forest building.
    min_samples_leaf : int or float (default = 1)
        The minimum number of samples (rows) in each leaf node.\n
         * If type ``int``, then ``min_samples_leaf`` represents the minimum
           number.
         * If ``float``, then ``min_samples_leaf`` represents a fraction and
           ``ceil(min_samples_leaf * n_rows)`` is the minimum number of
           samples for each leaf node.
    min_samples_split : int or float (default = 2)
        The minimum number of samples required to split an internal node.\n
         * If type ``int``, then min_samples_split represents the minimum
           number.
         * If type ``float``, then ``min_samples_split`` represents a fraction
           and ``max(2, ceil(min_samples_split * n_rows))`` is the minimum
           number of samples for each split.
    min_impurity_decrease : float (default = 0.0)
        Minimum decrease in impurity required for
        node to be split.
    max_batch_size : int (default = 4096)
        Maximum number of nodes that can be processed in a given batch.
    random_state : int (default = None)
        Seed for the random number generator. Unseeded by default.
    oob_score : bool (default = False)
        Whether to use out-of-bag samples to estimate the generalization
        accuracy. Only available if ``bootstrap=True``. The out-of-bag estimate
        provides a way to evaluate the model without requiring a separate
        validation set. The OOB score is computed using accuracy.
    class_weight : dict, 'balanced', or None, default=None
        Weights associated with classes. If ``'balanced'``, class weights are
        computed from the training labels.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    classes_ : np.ndarray, shape=(n_classes,)
        A sorted array of the class labels.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.
    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        ``oob_decision_function_`` might contain NaN. This attribute exists
        only when ``oob_score`` is True.
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.

    Notes
    -----
    While training the model for multi class classification problems, using
    deep trees or `max_features=1.0` provides better performance.

    For additional docs, see `scikitlearn's RandomForestClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_.

    When converting to sklearn using `as_sklearn()`, the `feature_importances_` attribute will return
    NaN values. If you need feature importances, save them before conversion:
    `importances = cuml_model.feature_importances_`
    """

    oob_decision_function_ = ReflectedAttr()

    _cpu_class_path = "sklearn.ensemble.RandomForestClassifier"

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "class_weight"]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.class_weight == "balanced_subsample":
            raise UnsupportedOnGPU(
                "`class_weight='balanced_subsample'` is not supported"
            )
        return {
            "class_weight": model.class_weight,
            **super()._params_from_cpu(model),
        }

    def _params_to_cpu(self):
        return {
            **super()._params_to_cpu(),
            "class_weight": self.class_weight,
        }

    def _attrs_from_cpu(self, model):
        return {
            "classes_": model.classes_,
            "n_classes_": model.n_classes_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        attrs = super()._attrs_to_cpu(model)
        # sklearn stores original labels on the forest and encoded labels on
        # each child tree.
        estimator_classes = np.arange(self.n_classes_, dtype=np.float64)
        for estimator in attrs.get("estimators_", ()):
            estimator.classes_ = estimator_classes
            estimator.n_classes_ = self.n_classes_
        return {
            **attrs,
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
        }

    def __init__(
        self,
        *,
        n_estimators=100,
        split_criterion="gini",
        bootstrap=True,
        max_samples=1.0,
        max_depth=None,
        max_leaves=-1,
        max_features="sqrt",
        n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        max_batch_size=4096,
        random_state=None,
        n_streams=4,
        oob_score=False,
        class_weight=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            split_criterion=split_criterion,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_samples=max_samples,
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_features=max_features,
            n_bins=n_bins,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_batch_size=max_batch_size,
            random_state=random_state,
            n_streams=n_streams,
            oob_score=oob_score,
            verbose=verbose,
            output_type=output_type,
        )
        self.class_weight = class_weight

    @nvtx.annotate(
        message="fit RF-Classifier @randomforestclassifier.pyx",
        domain="cuml_python",
    )
    @generate_docstring(y="dense_intdtype")
    @mlfunc(set_input_type=True)
    def fit(self, X, y, sample_weight=None) -> "RandomForestClassifier":
        """
        Perform Random Forest Classification on the input data
        """
        X, y, sample_weight, classes = check_inputs(
            self,
            X,
            y,
            sample_weight,
            dtype=("float32", "float64"),
            order="A",
            y_dtype="int32",
            sample_weight_dtype="float64",
            return_classes=True,
            reset=True,
        )
        self.classes_ = classes
        self.n_classes_ = len(classes)
        _, sample_weight = process_class_weight(
            classes,
            y,
            class_weight=self.class_weight,
            sample_weight=sample_weight,
            dtype=np.float64,
        )
        return self._fit_forest(X, y, sample_weight=sample_weight)

    @nvtx.annotate(
        message="predict RF-Classifier @randomforestclassifier.pyx",
        domain="cuml_python",
    )
    @insert_into_docstring(
        parameters=[("dense", "(n_samples, n_features)")],
        return_values=[("dense", "(n_samples, 1)")],
    )
    @mlfunc(preserve_index=True)
    def predict(
        self,
        X,
        *,
        threshold=0.5,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : {}
        threshold : float (default = 0.5)
            Threshold used for classification.
        layout : string (default = 'depth_first')
            Forest layout for GPU inference. Options: 'depth_first', 'layered',
            'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Controls batch subdivision for parallel processing. Optimal value depends
            on hardware, model and batch size. If None, determined automatically.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded to this byte alignment, which can
            improve performance. Typical values are 0 or 128 on GPU.

        Returns
        -------
        y : {}
        """
        nvforest_model = self._get_inference_nvforest_model(
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        X = check_inputs(
            self,
            X,
            dtype=nvforest_model.forest.get_dtype(),
            order="C",
            mem_type="device",
        )
        indices = nvforest_model.predict(X, threshold=threshold)
        return ClassLabels(indices, self.classes_)

    @insert_into_docstring(
        parameters=[("dense", "(n_samples, n_features)")],
        return_values=[("dense", "(n_samples, 1)")],
    )
    @mlfunc(preserve_index=True)
    def predict_proba(
        self,
        X,
        *,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        """
        Predicts class probabilities for X.

        Parameters
        ----------
        X : {}
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        y : {}
        """
        nvforest_model = self._get_inference_nvforest_model(
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        X = check_inputs(
            self,
            X,
            dtype=nvforest_model.forest.get_dtype(),
            order="C",
            mem_type="device",
        )
        return nvforest_model.predict_proba(X)

    @insert_into_docstring(
        parameters=[("dense", "(n_samples, n_features)")],
        return_values=[("dense", "(n_samples, 1)")],
    )
    @mlfunc(preserve_index=True)
    def predict_log_proba(
        self,
        X,
        *,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        """
        Predicts log class probabilities for X.

        Parameters
        ----------
        X : {}
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        y : {}
        """
        out = self.predict_proba(
            X,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        cp.log(out, out=out)
        return out

    @nvtx.annotate(
        message="score RF-Classifier @randomforestclassifier.pyx",
        domain="cuml_python",
    )
    @insert_into_docstring(
        parameters=[
            ("dense", "(n_samples, n_features)"),
            ("dense_intdtype", "(n_samples, 1)"),
        ]
    )
    @mlfunc(convert_output=False)
    def score(
        self,
        X,
        y,
        sample_weight=None,
        *,
        threshold=0.5,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        """
        Calculates the accuracy score of the model on test data.

        Parameters
        ----------
        X : {}
        y : {}
        sample_weight : array-like, shape=(n_samples,), default=None
            Sample weights for weighted mean accuracy.
        threshold : float (default = 0.5)
            Threshold used for classification predictions
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        accuracy : float
           Accuracy of the model [0.0 - 1.0]
        """
        return super().score(
            X,
            y,
            sample_weight=sample_weight,
            threshold=threshold,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
