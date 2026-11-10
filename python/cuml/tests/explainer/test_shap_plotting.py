#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

from cuml import KernelExplainer as cuKE
from cuml import LinearRegression

# set this variable to True if you want to see the charts
show_plots = False


def get_shap_values(
    explainer, dataset, train_or_test="test", api_type="raw_shap_values"
):
    X_test, X_train, _, _ = dataset
    if train_or_test == "test":
        explained_dataset = X_test
    elif train_or_test == "train":
        explained_dataset = X_train

    if api_type == "raw_shap_values":
        shap_values = explainer.shap_values(explained_dataset)
    elif api_type == "explanation_object":
        shap_values = explainer(explained_dataset)

    return shap_values, explained_dataset


@pytest.fixture(scope="module")
def explainer(exact_shap_regression_dataset):
    X_train, X_test, y_train, y_test = exact_shap_regression_dataset

    mod = LinearRegression().fit(X_train, y_train)

    explainer = cuKE(model=mod.predict, data=X_train)

    return explainer


def test_bar(explainer, exact_shap_regression_dataset):
    shap = pytest.importorskip("shap")
    shap_values, _ = get_shap_values(
        explainer=explainer,
        dataset=exact_shap_regression_dataset,
        api_type="explanation_object",
    )

    shap.plots.bar(shap_values, show=show_plots)


def test_decision_plot(explainer, exact_shap_regression_dataset):
    shap = pytest.importorskip("shap")
    shap_values, _ = get_shap_values(
        explainer=explainer,
        dataset=exact_shap_regression_dataset,
        api_type="raw_shap_values",
    )

    shap.decision_plot(0, shap_values, show=show_plots)


def test_dependence_plot(explainer, exact_shap_regression_dataset):
    shap = pytest.importorskip("shap")
    shap_values, data = get_shap_values(
        explainer=explainer,
        dataset=exact_shap_regression_dataset,
        train_or_test="train",
        api_type="raw_shap_values",
    )

    shap.dependence_plot(0, shap_values, data, show=show_plots)


def test_heatmap(explainer, exact_shap_regression_dataset):
    shap = pytest.importorskip("shap")
    shap_values, _ = get_shap_values(
        explainer=explainer,
        dataset=exact_shap_regression_dataset,
        api_type="explanation_object",
    )

    shap.plots.heatmap(shap_values, show=show_plots)


def test_summary(explainer, exact_shap_regression_dataset):
    """Check that the summary plot is unchanged."""
    shap = pytest.importorskip("shap")
    shap_values, _ = get_shap_values(
        explainer=explainer,
        dataset=exact_shap_regression_dataset,
        train_or_test="train",
        api_type="raw_shap_values",
    )

    shap.summary_plot(
        shap_values, show=show_plots, rng=np.random.default_rng(42)
    )


def test_violin(explainer, exact_shap_regression_dataset):
    """Check that the violin plot is unchanged."""
    shap = pytest.importorskip("shap")
    shap_values, _ = get_shap_values(
        explainer=explainer,
        dataset=exact_shap_regression_dataset,
        train_or_test="train",
        api_type="raw_shap_values",
    )

    shap.plots.violin(shap_values, show=show_plots)


def test_waterfall(explainer, exact_shap_regression_dataset):
    """Check that the waterfall plot is unchanged."""
    shap = pytest.importorskip("shap")
    shap_values, _ = get_shap_values(
        explainer=explainer,
        dataset=exact_shap_regression_dataset,
        api_type="explanation_object",
    )

    shap.plots.waterfall(shap_values[0], show=show_plots)
