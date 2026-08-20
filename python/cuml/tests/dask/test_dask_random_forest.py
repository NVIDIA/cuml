# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import cudf
import cupy as cp
import dask
import dask_cudf
import numpy as np
import pandas as pd
import pytest
import treelite
from dask.array import from_array
from dask.distributed import Client
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from cuml.dask._compat import DASK_2025_4_0
from cuml.dask.common import utils as dask_utils
from cuml.dask.ensemble import RandomForestClassifier as cuRFC_mg
from cuml.dask.ensemble import RandomForestRegressor as cuRFR_mg
from cuml.ensemble import RandomForestClassifier as cuRFC_sg
from cuml.ensemble import RandomForestRegressor as cuRFR_sg


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, y_train_df = dask_utils.persist_across_workers(
        c, [X_train_df, y_train_df], workers=workers
    )
    return X_train_df, y_train_df


def _get_treelite_bytes(model):
    return model._treelite_model_bytes


def _get_rank_local_oob_score(model):
    return model.oob_score_


def _get_rank_local_oob_prediction(model):
    if model._estimator_type == "regressor":
        return model.oob_prediction_
    return model.oob_decision_function_


@pytest.mark.parametrize("partitions_per_worker", [3])
def test_rf_classification_multi_class(partitions_per_worker, cluster):
    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    c = Client(cluster)
    kwargs = {"n_workers": -1} if DASK_2025_4_0() else {}
    n_workers = len(c.scheduler_info(**kwargs)["workers"])

    try:
        X, y = make_classification(
            n_samples=n_workers * 8000,
            n_features=20,
            n_clusters_per_class=1,
            n_informative=10,
            random_state=123,
            n_classes=10,
        )

        X = X.astype(np.float32)
        y = y.astype(np.int32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=n_workers * 500, random_state=123
        )

        cu_rf_params = {
            "n_estimators": 25,
            "max_depth": 16,
            "n_bins": 256,
            "random_state": 10,
        }

        X_train_df, y_train_df = _prep_training_data(
            c, X_train, y_train, partitions_per_worker
        )

        cuml_mod = cuRFC_mg(**cu_rf_params)
        cuml_mod.fit(X_train_df, y_train_df)
        X_test_dask_array = from_array(X_test)
        cuml_preds_gpu = cuml_mod.predict(X_test_dask_array).compute()
        acc_score_gpu = accuracy_score(cuml_preds_gpu, y_test)

        # Compare with sklearn baseline
        sk_model = skrfc(
            n_estimators=cu_rf_params["n_estimators"],
            max_depth=cu_rf_params["max_depth"],
            random_state=cu_rf_params["random_state"],
            n_jobs=-1,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)

        # Observed: mean=0.002, range=[0.002, 0.002], stderr=0.000
        assert acc_score_gpu >= (sk_acc - 0.07)

    finally:
        c.close()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_regression_dask_nvforest(partitions_per_worker, dtype, client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    # Use CUDA_VISIBLE_DEVICES to control the number of workers
    X, y = make_regression(
        n_samples=n_workers * 4000,
        n_features=20,
        n_informative=10,
        random_state=123,
    )

    X = X.astype(dtype)
    y = y.astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 100, random_state=123
    )

    cu_rf_params = {
        "n_estimators": 50,
        "max_depth": 16,
        "n_bins": 16,
    }

    workers = client.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)

    X_cudf = cudf.DataFrame(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = cudf.Series(y_train)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)
    X_cudf_test = cudf.DataFrame(pd.DataFrame(X_test))
    X_test_df = dask_cudf.from_cudf(X_cudf_test, npartitions=n_partitions)

    cuml_mod = cuRFR_mg(**cu_rf_params)
    cuml_mod.fit(X_train_df, y_train_df)

    cuml_mod_predict = cuml_mod.predict(X_test_df)
    cuml_mod_predict = cp.asnumpy(cp.array(cuml_mod_predict.compute()))

    acc_score = r2_score(y_test, cuml_mod_predict)

    assert acc_score >= 0.59


def test_rf_regression_nan_on_one_worker(client):
    workers = list(client.scheduler_info(n_workers=-1)["workers"])
    if len(workers) < 2:
        pytest.skip("This test requires at least two workers")

    X_parts = []
    y_parts = []
    for rank, worker in enumerate(workers):
        X_part = cudf.DataFrame(
            np.arange(80, dtype=np.float32).reshape(20, 4) + rank
        )
        y_part = cudf.Series(np.arange(20, dtype=np.float32) + rank)
        if rank == 0:
            X_part.iloc[0, 0] = np.nan

        X_parts.append(client.scatter(X_part, workers=[worker]))
        y_parts.append(client.scatter(y_part, workers=[worker]))

    X = dask_cudf.from_delayed(X_parts, meta=X_part.iloc[:0])
    y = dask_cudf.from_delayed(y_parts, meta=y_part.iloc[:0])

    model = cuRFR_mg(n_estimators=5, max_depth=3)
    with pytest.raises(RuntimeError, match="Input X contains NaN"):
        model.fit(X, y)


@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_classification_dask_array(partitions_per_worker, client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    X, y = make_classification(
        n_samples=n_workers * 2000,
        n_features=30,
        n_clusters_per_class=1,
        n_informative=20,
        random_state=123,
        n_classes=2,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 400
    )

    cu_rf_params = {
        "n_estimators": 25,
        "max_depth": 13,
        "n_bins": 15,
    }

    X_train_df, y_train_df = _prep_training_data(
        client, X_train, y_train, partitions_per_worker
    )
    X_test_dask_array = from_array(X_test)
    cuml_mod = cuRFC_mg(**cu_rf_params)
    cuml_mod.fit(X_train_df, y_train_df)
    cuml_mod_predict = cuml_mod.predict(X_test_dask_array).compute()

    acc_score = accuracy_score(cuml_mod_predict, y_test, normalize=True)

    assert acc_score > 0.8


@pytest.mark.parametrize("partitions_per_worker", [5])
def test_rf_classification_dask_nvforest_predict_proba(
    partitions_per_worker, client
):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    X, y = make_classification(
        n_samples=n_workers * 1500,
        n_features=30,
        n_clusters_per_class=1,
        n_informative=20,
        random_state=123,
        n_classes=2,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_workers * 150, random_state=123
    )

    cu_rf_params = {
        "n_bins": 16,
        "n_estimators": 40,
        "max_depth": 16,
    }

    X_train_df, y_train_df = _prep_training_data(
        client, X_train, y_train, partitions_per_worker
    )
    X_test_df, _ = _prep_training_data(
        client, X_test, y_test, partitions_per_worker
    )
    cu_rf_mg = cuRFC_mg(**cu_rf_params)
    cu_rf_mg.fit(X_train_df, y_train_df)

    nvforest_preds = cu_rf_mg.predict(X_test_df).compute()
    nvforest_preds = nvforest_preds.to_numpy()
    nvforest_preds_proba = cu_rf_mg.predict_proba(X_test_df).compute()
    nvforest_preds_proba = nvforest_preds_proba.to_numpy()
    np.testing.assert_equal(
        nvforest_preds, np.argmax(nvforest_preds_proba, axis=1)
    )

    y_proba = np.zeros(np.shape(nvforest_preds_proba))
    y_proba[:, 1] = y_test
    y_proba[:, 0] = 1.0 - y_test
    nvforest_mse = mean_squared_error(y_proba, nvforest_preds_proba)
    sk_model = skrfc(
        n_estimators=cu_rf_params["n_estimators"],
        max_depth=cu_rf_params["max_depth"],
        random_state=10,
    )
    sk_model.fit(X_train, y_train)
    sk_preds_proba = sk_model.predict_proba(X_test)
    sk_mse = mean_squared_error(y_proba, sk_preds_proba)

    # The threshold is required as the test would intermitently
    # fail with a max difference of 0.029 between the two mse values
    assert nvforest_mse <= sk_mse + 0.029


@pytest.mark.parametrize("model_type", ["classification", "regression"])
def test_rf_distributed_model(client, model_type):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])

    X, y = make_classification(
        n_samples=n_workers * 200, n_features=30, random_state=123, n_classes=2
    )

    X = X.astype(np.float32)
    if model_type == "classification":
        y = y.astype(np.int32)
    else:
        y = y.astype(np.float32)
    n_estimators = 40
    cu_rf_params = {"n_estimators": n_estimators, "max_depth": 16}

    X_df, y_df = _prep_training_data(client, X, y, partitions_per_worker=2)

    if model_type == "classification":
        cu_rf_mg = cuRFC_mg(**cu_rf_params)
    else:
        cu_rf_mg = cuRFR_mg(**cu_rf_params)

    cu_rf_mg.fit(X_df, y_df)
    model = cu_rf_mg.get_combined_model()
    treelite_bytes = model._treelite_model_bytes
    local_tl = treelite.Model.deserialize_bytes(treelite_bytes)
    assert local_tl.num_tree == n_estimators
    worker_model_bytes = client.gather(
        [
            client.submit(_get_treelite_bytes, model, workers=[worker])
            for worker, model in cu_rf_mg.rfs.items()
        ]
    )
    assert all(data == worker_model_bytes[0] for data in worker_model_bytes)


def test_rf_classification_uses_global_classes(client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])
    if n_workers < 2:
        pytest.skip("This test requires at least two workers")

    rows_per_worker = 100
    y = np.repeat(np.arange(n_workers), rows_per_worker).astype(np.int32)
    X = np.column_stack((y, np.arange(y.size))).astype(np.float32)
    X_dask, y_dask = _prep_training_data(client, X, y, partitions_per_worker=1)

    model = cuRFC_mg(
        n_estimators=1,
        bootstrap=False,
        max_depth=4,
        n_bins=max(2, n_workers),
        random_state=42,
    ).fit(X_dask, y_dask)

    np.testing.assert_array_equal(
        model.get_combined_model().classes_, np.arange(n_workers)
    )


def test_rf_classification_balanced_class_weight(client):
    """
    Ensure that class_weight='balanced' uses global class distributions.
    Test the functionality with class-segregated partitions.
    """
    workers = list(client.scheduler_info(n_workers=-1)["workers"])[:2]
    if len(workers) < 2:
        pytest.skip("This test requires at least two workers")

    def distributed_data(y_parts):
        X_partitions = []
        y_partitions = []
        for y_part, worker in zip(y_parts, workers):
            X_part = np.zeros((len(y_part), 1), dtype=np.float32)
            X_future = client.scatter(
                cudf.DataFrame(X_part), workers=[worker], hash=False
            )
            y_future = client.scatter(
                cudf.Series(y_part), workers=[worker], hash=False
            )
            with dask.annotate(workers=[worker]):
                X_partitions.append(
                    dask_cudf.from_delayed(
                        [X_future], meta=cudf.DataFrame(X_part).iloc[:0]
                    )
                )
                y_partitions.append(
                    dask_cudf.from_delayed(
                        [y_future], meta=cudf.Series(y_part).iloc[:0]
                    )
                )
        return (
            dask_cudf.concat(X_partitions),
            dask_cudf.concat(y_partitions),
        )

    # Both layouts contain the same imbalanced dataset (24 zeros and 8 ones).
    # The constant feature isolates differences in the weighted leaf values.
    labels_per_worker = np.array([0] * 12 + [1] * 4, dtype=np.int32)
    X_even, y_even = distributed_data([labels_per_worker, labels_per_worker])
    X_segregated, y_segregated = distributed_data(
        [
            np.ones(8, dtype=np.int32),
            np.zeros(24, dtype=np.int32),
        ]
    )

    params = {
        "workers": workers,
        "n_estimators": 1,
        "bootstrap": False,
        "max_depth": 1,
        "max_features": 1.0,
        "n_bins": 2,
        "random_state": 42,
        "class_weight": "balanced",
    }
    with dask.annotate(workers=workers):
        even_model = cuRFC_mg(**params).fit(X_even, y_even)
        segregated_model = cuRFC_mg(**params).fit(X_segregated, y_segregated)

    assert (
        even_model.get_combined_model()._treelite_model_bytes
        == segregated_model.get_combined_model()._treelite_model_bytes
    )


@pytest.mark.parametrize("mode", ["classification", "regression"])
def test_random_forest_oob_score(client, mode):
    """
    Ensure that Out-of-bag (OOB) scoring is correct.
    Distributed RF must perform an all-reduce over OOB estimates from each rank.
    Test: The first rank gets an easy dataset (X, y), while the second rank
          gets a noisy dataset (X, y). Two ranks will perform all-reduce to
          obtain the global OOB score.
    """
    workers = list(client.scheduler_info(n_workers=-1)["workers"])[:2]
    if len(workers) < 2:
        pytest.skip("This test requires at least two workers")

    n_samples_per_worker = 500
    if mode == "classification":
        X_easy, y_easy = make_classification(
            n_samples=n_samples_per_worker,
            n_features=10,
            n_informative=8,
            n_redundant=0,
            class_sep=3.0,
            flip_y=0.0,
            random_state=42,
        )
        model_cls = cuRFC_mg
        y_dtype = np.int32
    else:
        X_easy, y_easy = make_regression(
            n_samples=n_samples_per_worker,
            n_features=10,
            n_informative=8,
            random_state=42,
        )
        model_cls = cuRFR_mg
        y_dtype = np.float32

    rng = np.random.default_rng(42)
    X_noisy = rng.standard_normal((n_samples_per_worker, 10))
    if mode == "classification":
        y_noisy = rng.integers(0, 2, n_samples_per_worker)
    else:
        y_noisy = rng.uniform(0, 2, n_samples_per_worker)

    X_partitions = []
    y_partitions = []
    for worker, X_part, y_part in zip(
        workers, [X_easy, X_noisy], [y_easy, y_noisy]
    ):
        X_part = X_part.astype(np.float32)
        y_part = y_part.astype(y_dtype)
        X_future = client.scatter(
            cudf.DataFrame(X_part), workers=[worker], hash=False
        )
        y_future = client.scatter(
            cudf.Series(y_part), workers=[worker], hash=False
        )
        with dask.annotate(workers=[worker]):
            X_partitions.append(
                dask_cudf.from_delayed(
                    [X_future], meta=cudf.DataFrame(X_part).iloc[:0]
                )
            )
            y_partitions.append(
                dask_cudf.from_delayed(
                    [y_future], meta=cudf.Series(y_part).iloc[:0]
                )
            )

    X = dask_cudf.concat(X_partitions)
    y = dask_cudf.concat(y_partitions)
    with dask.annotate(workers=workers):
        model = model_cls(
            workers=workers,
            n_estimators=50,
            bootstrap=True,
            oob_score=True,
            max_samples=0.8,
            max_depth=10,
            random_state=42,
        ).fit(X, y)

    rank_oob_scores = client.gather(
        [
            client.submit(
                _get_rank_local_oob_score,
                model.rfs[worker],
                workers=[worker],
            )
            for worker in workers
        ]
    )
    assert model.oob_score_ == pytest.approx(rank_oob_scores[0])
    assert model.oob_score_ == pytest.approx(rank_oob_scores[1])

    rank_oob_preds = client.gather(
        [
            client.submit(
                _get_rank_local_oob_prediction,
                model.rfs[worker],
                workers=[worker],
            )
            for worker in workers
        ]
    )
    if mode == "regression":
        np.testing.assert_almost_equal(
            model.oob_prediction_.to_numpy(), rank_oob_preds[0].to_numpy()
        )
        np.testing.assert_almost_equal(
            model.oob_prediction_.to_numpy(), rank_oob_preds[1].to_numpy()
        )
    else:
        np.testing.assert_almost_equal(
            model.oob_decision_function_.to_numpy(),
            rank_oob_preds[0].to_numpy(),
        )
        np.testing.assert_almost_equal(
            model.oob_decision_function_.to_numpy(),
            rank_oob_preds[1].to_numpy(),
        )


def test_single_input_regression(client):
    X, y = make_classification(n_samples=1, n_classes=1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X, y = _prep_training_data(client, X, y, partitions_per_worker=2)
    cu_rf_mg = cuRFR_mg(n_bins=1)
    cu_rf_mg.fit(X, y)
    cuml_mod_predict = cu_rf_mg.predict(X)
    cuml_mod_predict = cp.asnumpy(cp.array(cuml_mod_predict.compute()))
    y = cp.asnumpy(cp.array(y.compute()))
    assert y[0] == cuml_mod_predict[0]


@pytest.mark.parametrize("max_depth", [1, 2, 3, 5, 10, 15, 20])
@pytest.mark.parametrize("n_estimators", [1, 5, 10, 20])
def test_rf_data_count(client, max_depth, n_estimators):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])
    n_samples_per_worker = 350

    X, y = make_classification(
        n_samples=n_samples_per_worker * n_workers,
        n_features=20,
        n_clusters_per_class=1,
        n_informative=10,
        random_state=123,
        n_classes=2,
    )
    X = X.astype(np.float32)
    dask_model = cuRFC_mg(
        max_features=1.0,
        max_samples=1.0,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=23707,
        n_estimators=n_estimators,
        max_leaves=-1,
        max_depth=max_depth,
    )
    y = y.astype(np.int32)

    X_dask, y_dask = _prep_training_data(client, X, y, partitions_per_worker=2)
    dask_model.fit(X_dask, y_dask)
    model = dask_model.get_combined_model()
    json_obj = json.loads(model.as_treelite().dump_as_json())

    def check_count(node, nodes):
        if "left_child" in node:
            left = nodes[node["left_child"]]
            right = nodes[node["right_child"]]
            count = check_count(left, nodes) + check_count(right, nodes)
            assert count == node["data_count"]
        return node["data_count"]

    for tree in json_obj["trees"]:
        nodes = tree["nodes"]
        # The root contains rows from the complete distributed dataset.
        assert nodes[0]["data_count"] == n_samples_per_worker * n_workers
        # Check that the data_count accumulates properly as you move up the tree
        for node in nodes:
            check_count(node, nodes)


def test_unlimited_max_depth_classifier(client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])
    X, y = make_classification(
        n_samples=n_workers * 200, n_features=10, random_state=42
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_dask, y_dask = _prep_training_data(client, X, y, partitions_per_worker=1)
    clf = cuRFC_mg(n_estimators=5, max_depth=None)
    clf.fit(X_dask, y_dask)
    preds = cp.asnumpy(cp.array(clf.predict(X_dask).compute()))
    assert len(preds) == len(y)


def test_unlimited_max_depth_regressor(client):
    n_workers = len(client.scheduler_info(n_workers=-1)["workers"])
    X, y = make_regression(
        n_samples=n_workers * 200, n_features=10, random_state=42
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_dask, y_dask = _prep_training_data(client, X, y, partitions_per_worker=1)
    reg = cuRFR_mg(n_estimators=5, max_depth=None)
    reg.fit(X_dask, y_dask)
    preds = cp.asnumpy(cp.array(reg.predict(X_dask).compute()))
    assert len(preds) == len(y)


@pytest.mark.parametrize("estimator_type", ["regression", "classification"])
def test_rf_get_model_right_after_fit(client, estimator_type):
    max_depth = 3
    n_estimators = 5

    X, y = make_classification()
    X = X.astype(np.float32)
    if estimator_type == "classification":
        cu_rf_mg = cuRFC_mg(
            max_features=1.0,
            max_samples=1.0,
            n_bins=16,
            n_estimators=n_estimators,
            max_leaves=-1,
            max_depth=max_depth,
        )
        y = y.astype(np.int32)
    elif estimator_type == "regression":
        cu_rf_mg = cuRFR_mg(
            max_features=1.0,
            max_samples=1.0,
            n_bins=16,
            n_estimators=n_estimators,
            max_leaves=-1,
            max_depth=max_depth,
        )
        y = y.astype(np.float32)
    else:
        assert False
    X_dask, y_dask = _prep_training_data(client, X, y, partitions_per_worker=2)
    cu_rf_mg.fit(X_dask, y_dask)
    single_gpu_model = cu_rf_mg.get_combined_model()
    if estimator_type == "classification":
        assert isinstance(single_gpu_model, cuRFC_sg)
    elif estimator_type == "regression":
        assert isinstance(single_gpu_model, cuRFR_sg)
    else:
        assert False
