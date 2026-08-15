#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings
from dataclasses import dataclass, field

from sklearn.utils import (
    ClassifierTags,
    InputTags,
    RegressorTags,
    Tags,
    TargetTags,
    TransformerTags,
)

from cuml.common.doc_utils import generate_docstring
from cuml.internals.outputs import ClassLabels, mlfunc


class DeprecatedGetFeatureNamesMixin:
    """Defines the deprecated `get_feature_names` method."""

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        .. deprecated:: 26.10

            This method was deprecated in version 26.10 and will be removed in
            version 26.12. Please use `get_feature_names_out` instead.
        """
        warnings.warn(
            "`get_feature_names` was deprecated in version 26.10 and will be "
            "removed in version 26.12. Please use `get_feature_names_out` instead",
            FutureWarning,
        )
        return self.get_feature_names_out(
            input_features=input_features
        ).tolist()


###############################################################################
#                          Tag Functionality Mixin                            #
###############################################################################


@dataclass(slots=True)
class CumlTags(Tags):
    preferred_input_order: str | None = None
    X_types_gpu: list[str] = field(default_factory=lambda: ["2darray"])


def _ensure_transformer_tags(tags):
    if tags.transformer_tags is None:
        tags.transformer_tags = TransformerTags()
    return tags.transformer_tags


class TagsMixin:
    """Chain terminator for cuML's ``__sklearn_tags__`` super-chain.

    Must be the last (right-most) tag-providing ancestor in MRO. Subclasses
    that want to contribute tags should override ``__sklearn_tags__`` by
    calling ``super().__sklearn_tags__()``, mutating the returned object, and
    returning it.
    """

    def __sklearn_tags__(self):
        return CumlTags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            input_tags=InputTags(two_d_array=True),
        )


###############################################################################
#                          Estimator Type Mixins                              #
#                 Estimators should only use one of these.                    #
###############################################################################


class RegressorMixin:
    """
    Mixin class for regression estimators in cuML
    """

    _estimator_type = "regressor"

    @generate_docstring(
        return_values={
            "name": "score",
            "type": "float",
            "description": "R^2 of self.predict(X) wrt. y.",
        }
    )
    @mlfunc(convert_output=False)
    def score(self, X, y, sample_weight=None, **kwargs):
        """
        Scoring function for regression estimators

        Returns the coefficient of determination R^2 of the prediction.

        """
        from cuml.metrics.regression import r2_score

        preds = self.predict(X, **kwargs)
        return r2_score(y, preds, sample_weight=sample_weight)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.target_tags.required = True
        tags.regressor_tags = RegressorTags()
        return tags


class ClassifierMixin:
    """
    Mixin class for classifier estimators in cuML
    """

    _estimator_type = "classifier"

    @generate_docstring(
        return_values={
            "name": "score",
            "type": "float",
            "description": (
                "Accuracy of self.predict(X) wrt. y "
                "(fraction where y == pred_y)"
            ),
        }
    )
    @mlfunc(convert_output=False)
    def score(self, X, y, sample_weight=None, **kwargs):
        """
        Scoring function for classifier estimators based on mean accuracy.

        """
        from cuml.metrics import accuracy_score

        preds = self.predict(X, **kwargs)
        if isinstance(preds, ClassLabels):
            preds = preds.to_output()
        return accuracy_score(y, preds, sample_weight=sample_weight)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.target_tags.required = True
        tags.classifier_tags = ClassifierTags()
        return tags


class ClusterMixin:
    """
    Mixin class for clustering estimators in cuML.
    """

    _estimator_type = "clusterer"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "clusterer"
        tags.target_tags.required = False
        return tags


###############################################################################
#                              Input Mixins                                   #
#               Estimators can use as many of these as needed.                #
###############################################################################


class FMajorInputTagMixin:
    """
    Mixin class for estimators that prefer inputs in F (column major) order.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.preferred_input_order = "F"
        return tags


class CMajorInputTagMixin:
    """
    Mixin class for estimators that prefer inputs in C (row major) order.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.preferred_input_order = "C"
        return tags


class SparseInputTagMixin:
    """
    Mixin class for estimators that can take (GPU and host) sparse inputs.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.X_types_gpu = ["2darray", "sparse"]
        return tags


class StringInputTagMixin:
    """
    Mixin class for estimators that can take (GPU and host) string inputs.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.X_types_gpu = ["2darray", "string"]
        return tags


class AllowNaNTagMixin:
    """
    Mixin class for estimators that allow NaNs in their inputs.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
