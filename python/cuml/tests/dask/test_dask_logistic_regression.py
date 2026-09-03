# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import random
import warnings
from contextlib import contextmanager

import cudf
import cupy as cp
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.metrics import accuracy_score

from cuml.dask.common import utils as dask_utils
from cuml.testing.utils import array_equal


@contextmanager
def ignore_deprecated_lbfgs_params_warning():
    """Ignores a warning in sklearn raised by scipy.optimize deprecated params"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        yield


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = np.array(pd.DataFrame(y_train).values)
    y_cudf = y_cudf[:, 0]
    y_cudf = cudf.Series(y_cudf)
    y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, y_train_df = dask_utils.persist_across_workers(
        c, [X_train_df, y_train_df], workers=workers
    )
    return X_train_df, y_train_df


def _prep_training_data_sparse(c, X_train, y_train, partitions_per_worker):
    assert isinstance(X_train, csr_matrix)
    "The implementation follows test_dask_tfidf.create_cp_sparse_dask_array"
    import dask.array as da

    workers = c.has_what().keys()
    target_n_partitions = partitions_per_worker * len(workers)

    def cal_chunks(dataset, n_partitions):
        n_samples = dataset.shape[0]
        # Avoid zero-row partitions when a small test dataset has fewer rows
        # than the number of partitions requested by the available workers.
        n_partitions = min(n_partitions, max(n_samples, 1))
        n_samples_per_part = int(n_samples / n_partitions)
        chunk_sizes = [n_samples_per_part] * n_partitions
        samples_last_row = n_samples - (
            (n_partitions - 1) * n_samples_per_part
        )
        chunk_sizes[-1] = samples_last_row
        return tuple(chunk_sizes)

    assert X_train.shape[0] == y_train.shape[0], (
        "the number of data records is not equal to the number of labels"
    )
    target_chunk_sizes = cal_chunks(X_train, target_n_partitions)

    X_da = da.from_array(X_train, chunks=(target_chunk_sizes, -1))
    y_da = da.from_array(y_train, chunks=target_chunk_sizes)

    # todo (dgd): Dask nightly packages break persisting
    # sparse arrays before using them.
    # https://github.com/NVIDIA/cuml/issues/6168

    # X_da, y_da = dask_utils.persist_across_workers(
    #     c, [X_da, y_da], workers=workers
    # )
    return X_da, y_da


def make_classification_dataset(
    datatype,
    nrows,
    ncols,
    n_info,
    n_redundant=2,
    n_classes=2,
    shift=0.0,
    scale=1.0,
):
    X, y = make_classification(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_redundant=n_redundant,
        n_classes=n_classes,
        shift=shift,
        scale=scale,
        random_state=0,
    )
    X = X.astype(datatype)
    y = y.astype(datatype)

    return X, y


@pytest.mark.mg
@pytest.mark.parametrize("n_parts", [2])
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_lbfgs_toy(n_parts, datatype, client):
    def imp():
        import cuml.comm.serialize  # NOQA

    client.run(imp)

    X = np.array([(1, 2), (1, 3), (2, 1), (3, 1)], datatype)
    y = np.array([1.0, 1.0, 0.0, 0.0], datatype)

    from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

    X_df, y_df = _prep_training_data(client, X, y, n_parts)

    lr = cumlLBFGS_dask()

    lr.fit(X_df, y_df)

    lr_coef = lr.coef_.to_numpy()
    lr_intercept = lr.intercept_.to_numpy()

    assert len(lr_coef) == 1
    assert lr_coef[0] == pytest.approx([-0.71483153, 0.7148315], abs=1e-6)
    assert lr_intercept == pytest.approx([-2.2614916e-08], abs=1e-6)

    # test predict
    preds = lr.predict(X_df, delayed=True).compute().to_numpy()
    from numpy.testing import assert_array_equal

    assert_array_equal(preds, y, strict=True)
    assert lr.dtype == datatype


def _test_lbfgs(
    nrows,
    ncols,
    n_parts,
    fit_intercept,
    datatype,
    delayed,
    client,
    penalty="l2",
    l1_ratio=None,
    C=1.0,
    standardization=False,
    n_classes=2,
    convert_to_sparse=False,
    _convert_index=False,
):
    tolerance = 0.01 if convert_to_sparse else 0.005

    def imp():
        import cuml.comm.serialize  # NOQA

    client.run(imp)

    from cuml.dask.linear_model.logistic_regression import (
        LogisticRegression as cumlLBFGS_dask,
    )

    # set n_informative variable for calling sklearn.datasets.make_classification
    n_info = 5
    nrows = int(nrows)
    ncols = int(ncols)
    X, y = make_classification_dataset(
        datatype, nrows, ncols, n_info, n_classes=n_classes
    )

    if convert_to_sparse:
        assert _convert_index == np.int32 or _convert_index == np.int64, (
            "only support np.int32 or np.int64 as index dtype"
        )
        X = csr_matrix(X)

        # X_dask and y_dask are dask array
        X_dask, y_dask = _prep_training_data_sparse(client, X, y, n_parts)
    else:
        # X_dask and y_dask are dask cudf
        X_dask, y_dask = _prep_training_data(client, X, y, n_parts)

    lr = cumlLBFGS_dask(
        solver="qn",
        fit_intercept=fit_intercept,
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
        standardization=standardization,
        _convert_index=_convert_index,
        verbose=True,
    )
    lr.fit(X_dask, y_dask)

    def array_to_numpy(ary):
        if isinstance(ary, cp.ndarray):
            return cp.asarray(ary)
        elif isinstance(ary, cudf.DataFrame) or isinstance(ary, cudf.Series):
            return ary.to_numpy()
        else:
            assert isinstance(ary, np.ndarray)
            return ary

    lr_coef = array_to_numpy(lr.coef_)
    lr_intercept = array_to_numpy(lr.intercept_)

    assert lr_coef.dtype == datatype
    assert lr_intercept.dtype == datatype

    if penalty == "l2" or penalty is None:
        sk_solver = "lbfgs"
    elif penalty == "l1" or penalty == "elasticnet":
        sk_solver = "saga"
    else:
        raise ValueError(f"unexpected penalty {penalty}")

    sk_model = skLR(
        solver=sk_solver,
        fit_intercept=fit_intercept,
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
    )
    with ignore_deprecated_lbfgs_params_warning():
        sk_model.fit(X, y)

    sk_coef = sk_model.coef_
    sk_intercept = sk_model.intercept_

    if sk_solver == "lbfgs" and standardization is False:
        assert len(lr_coef) == len(sk_coef)
        assert array_equal(
            lr_coef,
            sk_coef,
            unit_tol=tolerance,
            total_tol=tolerance,
            with_sign=True,
        )
        assert array_equal(
            lr_intercept,
            sk_intercept,
            unit_tol=tolerance,
            total_tol=tolerance,
            with_sign=True,
        )

    # test predict
    cu_preds = lr.predict(X_dask, delayed=delayed).compute()
    if isinstance(cu_preds, cp.ndarray):
        cu_preds = cp.asnumpy(cu_preds)
    if not isinstance(cu_preds, np.ndarray):
        cu_preds = cu_preds.to_numpy()
    accuracy_cuml = accuracy_score(y, cu_preds)

    sk_preds = sk_model.predict(X)
    accuracy_sk = accuracy_score(y, sk_preds)

    assert len(cu_preds) == len(sk_preds)
    assert (accuracy_cuml >= accuracy_sk) | (
        np.abs(accuracy_cuml - accuracy_sk) < 1e-3
    )

    return lr


@pytest.mark.mg
@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("delayed", [True, False])
def test_lbfgs(n_parts, fit_intercept, delayed, client):
    datatype = np.float32 if fit_intercept else np.float64

    lr = _test_lbfgs(
        nrows=1e5,
        ncols=20,
        n_parts=n_parts,
        fit_intercept=fit_intercept,
        datatype=datatype,
        delayed=delayed,
        client=client,
    )

    assert lr.dtype == datatype


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_noreg(fit_intercept, client):
    datatype = np.float64 if fit_intercept else np.float32
    lr = _test_lbfgs(
        nrows=1e5,
        ncols=20,
        n_parts=23,
        fit_intercept=fit_intercept,
        datatype=datatype,
        delayed=True,
        client=client,
        penalty=None,
    )

    l1_strength, l2_strength = lr._get_l1_l2_strength()
    assert l1_strength == 0.0
    assert l2_strength == 0.0

    assert lr.dtype == datatype


def test_n_classes_small(client):
    def assert_small(X, y, n_classes, check_predictions=False):
        X_df, y_df = _prep_training_data(client, X, y, partitions_per_worker=1)
        from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

        lr = cumlLBFGS_dask()
        lr.fit(X_df, y_df)
        assert len(lr.classes_) == n_classes
        if check_predictions:
            predictions = lr.predict(X_df, delayed=True).compute().to_numpy()
            assert np.array_equal(predictions, y)
        return lr

    X = np.array([(1, 2), (1, 3)], np.float32)
    y = np.array([1.0, 0.0], np.float32)
    lr = assert_small(X=X, y=y, n_classes=2)
    assert np.array_equal(lr.classes_, np.array([0.0, 1.0], np.float32))

    X = np.array([(1, 2), (1, 3), (1, 2.5)], np.float32)
    y = np.array([1.0, 0.0, 1.0], np.float32)
    lr = assert_small(X=X, y=y, n_classes=2)
    assert np.array_equal(lr.classes_, np.array([0.0, 1.0], np.float32))

    X = np.array([(1, 2), (1, 2.5), (1, 3)], np.float32)
    y = np.array([1.0, 1.0, 0.0], np.float32)
    lr = assert_small(X=X, y=y, n_classes=2)
    assert np.array_equal(lr.classes_, np.array([0.0, 1.0], np.float32))

    X = np.array([(1, 2), (1, 3), (1, 2.5)], np.float32)
    y = np.array([10.0, 50.0, 20.0], np.float32)
    lr = assert_small(X=X, y=y, n_classes=3, check_predictions=True)
    assert np.array_equal(
        lr.classes_, np.array([10.0, 20.0, 50.0], np.float32)
    )


@pytest.mark.parametrize("n_parts", [2, 23])
@pytest.mark.parametrize("n_classes", [2, 6])
def test_n_classes(n_parts, n_classes, client):
    nrows = int(1e5) if n_classes < 5 else int(2e5)
    lr = _test_lbfgs(
        nrows=nrows,
        ncols=20,
        n_parts=n_parts,
        fit_intercept=True,
        datatype=np.float32,
        delayed=True,
        client=client,
        penalty="l2",
        n_classes=n_classes,
    )

    assert len(lr.classes_) == n_classes
    assert lr.dtype == np.float32


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("delayed", [True])
@pytest.mark.parametrize("n_classes", [2, 6])
@pytest.mark.parametrize("C", [1.0, 10.0])
def test_l1(fit_intercept, delayed, n_classes, C, client):
    datatype = np.float64 if fit_intercept else np.float32
    nrows = int(1e5) if n_classes < 5 else int(2e5)
    lr = _test_lbfgs(
        nrows=nrows,
        ncols=20,
        n_parts=2,
        fit_intercept=fit_intercept,
        datatype=datatype,
        delayed=delayed,
        client=client,
        penalty="l1",
        n_classes=n_classes,
        C=C,
    )

    l1_strength, l2_strength = lr._get_l1_l2_strength()
    assert l1_strength == 1.0 / lr.C
    assert l2_strength == 0.0

    assert lr.dtype == datatype


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("delayed", [True])
@pytest.mark.parametrize("n_classes", [2, 6])
@pytest.mark.parametrize("l1_ratio", [0.2, 0.8])
def test_elasticnet(fit_intercept, delayed, n_classes, l1_ratio, client):
    datatype = np.float32 if fit_intercept else np.float64

    nrows = int(1e5) if n_classes < 5 else int(2e5)
    lr = _test_lbfgs(
        nrows=nrows,
        ncols=20,
        n_parts=2,
        fit_intercept=fit_intercept,
        datatype=datatype,
        delayed=delayed,
        client=client,
        penalty="elasticnet",
        n_classes=n_classes,
        l1_ratio=l1_ratio,
    )

    l1_strength, l2_strength = lr._get_l1_l2_strength()

    strength = 1.0 / lr.C
    assert l1_strength == lr.l1_ratio * strength
    assert l2_strength == (1.0 - lr.l1_ratio) * strength

    assert lr.dtype == datatype


@pytest.mark.mg
@pytest.mark.parametrize(
    "reg_dtype",
    [
        ((None, 1.0, None), np.float32),
        (("l2", 2.0, None), np.float64),
        (("l1", 2.0, None), np.float32),
        (("elasticnet", 2.0, 0.2), np.float64),
    ],
)
def test_sparse_from_dense(reg_dtype, client):
    penalty, C, l1_ratio = reg_dtype[0]
    datatype = reg_dtype[1]

    _convert_index = np.int32 if random.choice([True, False]) else np.int64

    lr = _test_lbfgs(
        nrows=int(1e5),
        ncols=20,
        n_parts=2,
        fit_intercept=True,
        datatype=datatype,
        delayed=True,
        client=client,
        penalty=penalty,
        n_classes=2,
        C=C,
        l1_ratio=l1_ratio,
        convert_to_sparse=True,
        _convert_index=_convert_index,
    )
    assert lr.dtype == datatype
    assert lr.index_dtype == _convert_index


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.filterwarnings(
    "ignore:The max_iter was reached which means the coef_ did not converge:sklearn.exceptions.ConvergenceWarning"
)
def test_sparse_nlp20news(dtype, sparse_text_dataset, client):
    X, y = sparse_text_dataset
    n_parts = 2  # partitions_per_worker

    from scipy.sparse import csr_matrix
    from sklearn.model_selection import train_test_split

    X = X.astype(dtype)

    X = csr_matrix(X)
    y = y.get().astype(dtype)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    from cuml.dask.linear_model import LogisticRegression as MG

    X_train_da, y_train_da = _prep_training_data_sparse(
        client, X_train, y_train, partitions_per_worker=n_parts
    )
    X_test_da, _ = _prep_training_data_sparse(
        client, X_test, y_test, partitions_per_worker=n_parts
    )

    cumg = MG(verbose=6, C=20.0)
    cumg.fit(X_train_da, y_train_da)

    preds = cumg.predict(X_test_da).compute()
    cuml_score = accuracy_score(y_test, preds.tolist())

    from sklearn.linear_model import LogisticRegression as CPULR

    cpu = CPULR(C=20.0, solver="saga")
    cpu.fit(X_train, y_train)
    cpu_preds = cpu.predict(X_test)
    cpu_score = accuracy_score(y_test, cpu_preds.tolist())
    assert cuml_score >= cpu_score or np.abs(cuml_score - cpu_score) < 1e-3


def test_exception_one_label(client):
    n_parts = 2
    datatype = "float32"

    X = np.array([(1, 2), (1, 3), (2, 1), (3, 1)], datatype)
    y = np.array([1.0, 1.0, 1.0, 1.0], datatype)
    X_df, y_df = _prep_training_data(client, X, y, n_parts)

    err_msg = r"loss='sigmoid' requires n_classes == 2 \(got 1\)"

    from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

    mg = cumlLBFGS_dask(fit_intercept=True, verbose=6)
    with pytest.raises(RuntimeError, match=err_msg):
        mg.fit(X_df, y_df)


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize(
    "reg_dtype",
    [
        ((None, 1.0, None), np.float64),
        (("l2", 2.0, None), np.float32),
        (("l1", 2.0, None), np.float64),
        (("elasticnet", 2.0, 0.2), np.float32),
    ],
)
@pytest.mark.parametrize("delayed", [False])
def test_standardization_on_normal_dataset(
    fit_intercept, reg_dtype, delayed, client
):
    regularization = reg_dtype[0]
    datatype = reg_dtype[1]
    penalty = regularization[0]
    C = regularization[1]
    l1_ratio = regularization[2]

    # test correctness compared with scikit-learn
    lr = _test_lbfgs(
        nrows=int(1e5),
        ncols=20,
        n_parts=2,
        fit_intercept=fit_intercept,
        datatype=datatype,
        delayed=delayed,
        client=client,
        penalty=penalty,
        n_classes=2,
        C=C,
        l1_ratio=l1_ratio,
        standardization=True,
    )
    assert lr.dtype == datatype


def standardize_dataset(X_train, X_test, fit_intercept):
    # This function is for testing standardization.
    # if fit_intercept is true, mean-center then scale the dataset
    # if fit_intercept is false, scale the dataset without mean-centering
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=fit_intercept, with_std=True)
    scaler.fit(X_train)
    scaler.scale_ = np.sqrt(scaler.var_ * len(X_train) / (len(X_train) - 1))

    def transform_func(scaler, X_data):
        X_res = scaler.transform(X_data)
        nan_mask = np.isnan(X_res)
        X_res[nan_mask] = X_data[nan_mask]
        return X_res

    X_train_scaled = transform_func(scaler, X_train)
    X_test_scaled = transform_func(scaler, X_test)
    return (X_train_scaled, X_test_scaled, scaler)


def adjust_standardization_model_for_comparison(
    coef_, intercept_, fit_intercept, scaler
):
    # This function is for testing standardization.
    # It converts the coef_ and intercept_ of Dask Cuml to align wih scikit-learn for comparison.
    coef_ = coef_ if isinstance(coef_, np.ndarray) else coef_.to_numpy()
    intercept_ = (
        intercept_
        if isinstance(intercept_, np.ndarray)
        else intercept_.to_numpy()
    )

    coef_origin = coef_ * scaler.scale_
    if fit_intercept is True:
        intercept_origin = intercept_ + np.dot(coef_, scaler.mean_)
    else:
        intercept_origin = intercept_
    return (coef_origin, intercept_origin)


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize(
    "reg_dtype",
    [
        ((None, 1.0, None), np.float32),
        (("l2", 2.0, None), np.float32),
        (("l1", 2.0, None), np.float64),
        (("elasticnet", 2.0, 0.2), np.float64),
    ],
)
@pytest.mark.parametrize("delayed", [False])
def test_standardization_on_scaled_dataset(
    fit_intercept, reg_dtype, delayed, client
):
    regularization = reg_dtype[0]
    datatype = reg_dtype[1]

    penalty = regularization[0]
    C = regularization[1]
    l1_ratio = regularization[2]
    nrows = int(2e5)
    ncols = 20
    n_classes = 6
    n_info = ncols
    n_redundant = 0
    n_parts = 2
    tolerance = 0.01

    from sklearn.linear_model import LogisticRegression as CPULR
    from sklearn.model_selection import train_test_split

    from cuml.dask.linear_model.logistic_regression import (
        LogisticRegression as cumlLBFGS_dask,
    )

    X, y = make_classification_dataset(
        datatype,
        nrows,
        ncols,
        n_info,
        n_redundant=n_redundant,
        n_classes=n_classes,
    )
    X[:, 0] *= 1000  # Scale up the first features by 1000
    X[:, 0] += 50  # Shift the first features by 50
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    def to_dask_data(X_train, X_test, y_train, y_test):
        X_train_dask, y_train_dask = _prep_training_data(
            client, X_train, y_train, n_parts
        )
        X_test_dask, y_test_dask = _prep_training_data(
            client, X_test, y_test, n_parts
        )
        return (X_train_dask, X_test_dask, y_train_dask, y_test_dask)

    # run MG with standardization=True
    mgon = cumlLBFGS_dask(
        standardization=True,
        solver="qn",
        fit_intercept=fit_intercept,
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
        verbose=True,
    )
    X_train_dask, X_test_dask, y_train_dask, _ = to_dask_data(
        X_train, X_test, y_train, y_test
    )
    mgon.fit(X_train_dask, y_train_dask)
    mgon_preds = (
        mgon.predict(X_test_dask, delayed=delayed).compute().to_numpy()
    )
    mgon_accuracy = accuracy_score(y_test, mgon_preds)

    assert array_equal(
        X_train_dask.compute().to_numpy(),
        X_train,
        unit_tol=tolerance,
        total_tol=tolerance,
    )

    # run CPU with standardized dataset
    X_train_scaled, X_test_scaled, scaler = standardize_dataset(
        X_train, X_test, fit_intercept
    )

    sk_solver = "lbfgs" if penalty == "l2" or penalty is None else "saga"
    cpu = CPULR(
        solver=sk_solver,
        fit_intercept=fit_intercept,
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
    )
    with ignore_deprecated_lbfgs_params_warning():
        cpu.fit(X_train_scaled, y_train)
    cpu_preds = cpu.predict(X_test_scaled)
    cpu_accuracy = accuracy_score(y_test, cpu_preds)

    assert len(mgon_preds) == len(cpu_preds)
    assert (mgon_accuracy >= cpu_accuracy) | (
        np.abs(mgon_accuracy - cpu_accuracy) < 1e-3
    )

    # assert equal the accuracy and the model
    (
        mgon_coef_origin,
        mgon_intercept_origin,
    ) = adjust_standardization_model_for_comparison(
        mgon.coef_, mgon.intercept_, fit_intercept, scaler
    )

    if sk_solver == "lbfgs":
        assert array_equal(
            mgon_coef_origin,
            cpu.coef_,
            unit_tol=tolerance,
            total_tol=tolerance,
        )
        assert array_equal(
            mgon_intercept_origin,
            cpu.intercept_,
            unit_tol=tolerance,
            total_tol=tolerance,
        )

    # running MG with standardization=False
    mgoff = cumlLBFGS_dask(
        standardization=False,
        solver="qn",
        fit_intercept=fit_intercept,
        penalty=penalty,
        l1_ratio=l1_ratio,
        C=C,
        verbose=True,
    )
    X_train_ds, X_test_ds, y_train_dask, _ = to_dask_data(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    mgoff.fit(X_train_ds, y_train_dask)
    mgoff_preds = (
        mgoff.predict(X_test_ds, delayed=delayed).compute().to_numpy()
    )
    mgoff_accuracy = accuracy_score(y_test, mgoff_preds)

    # assert equal the accuracy and the model
    assert len(mgon_preds) == len(mgoff_preds)
    assert (mgon_accuracy >= mgoff_accuracy) | (
        np.abs(mgon_accuracy - mgoff_accuracy) < 1e-3
    )

    assert array_equal(
        mgon_coef_origin,
        mgoff.coef_.to_numpy(),
        unit_tol=tolerance,
        total_tol=tolerance,
    )
    assert array_equal(
        mgon_intercept_origin,
        mgoff.intercept_.to_numpy(),
        unit_tol=tolerance,
        total_tol=tolerance,
    )

    assert mgon.dtype == datatype
    assert mgoff.dtype == datatype


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "reg_dtype",
    [
        ((None, 1.0, None), np.float64),
        (("l2", 2.0, None), np.float64),
        (("l1", 2.0, None), np.float32),
        (("elasticnet", 2.0, 0.2), np.float32),
    ],
)
def test_standardization_example(fit_intercept, reg_dtype, client):
    regularization = reg_dtype[0]
    datatype = reg_dtype[1]

    n_rows = int(1e5)
    n_cols = 20
    n_info = 10
    n_classes = 4

    n_parts = 2
    max_iter = 5  # cannot set this too large. Observed GPU-specific coefficients when objective converges at 0.

    penalty = regularization[0]
    C = regularization[1]
    l1_ratio = regularization[2]

    est_params = {
        "penalty": penalty,
        "C": C,
        "l1_ratio": l1_ratio,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
    }

    tolerance = 0.005

    X, y = make_classification_dataset(
        datatype, n_rows, n_cols, n_info, n_classes=n_classes
    )

    X_scaled, _, scaler = standardize_dataset(X, X, fit_intercept)

    X_df, y_df = _prep_training_data(client, X, y, n_parts)
    from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

    lr_on = cumlLBFGS_dask(standardization=True, verbose=True, **est_params)
    lr_on.fit(X_df, y_df)

    (
        lron_coef_origin,
        lron_intercept_origin,
    ) = adjust_standardization_model_for_comparison(
        lr_on.coef_, lr_on.intercept_, fit_intercept, scaler
    )

    X_df_scaled, y_df = _prep_training_data(client, X_scaled, y, n_parts)
    lr_off = cumlLBFGS_dask(standardization=False, **est_params)
    lr_off.fit(X_df_scaled, y_df)

    assert array_equal(
        lron_coef_origin,
        lr_off.coef_.to_numpy(),
        unit_tol=tolerance,
        total_tol=tolerance,
    )
    assert array_equal(
        lron_intercept_origin,
        lr_off.intercept_.to_numpy(),
        unit_tol=tolerance,
        total_tol=tolerance,
    )

    from cuml.linear_model import LogisticRegression as SG

    sg = SG(**est_params)
    sg.fit(X_scaled, y)

    assert array_equal(
        lron_coef_origin, sg.coef_, unit_tol=tolerance, total_tol=tolerance
    )
    assert array_equal(
        lron_intercept_origin,
        sg.intercept_,
        unit_tol=tolerance,
        total_tol=tolerance,
    )

    assert lr_on.dtype == datatype
    assert lr_off.dtype == datatype


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "reg_dtype",
    [
        ((None, 1.0, None), np.float64),
        (("l2", 2.0, None), np.float32),
        (("l1", 2.0, None), np.float64),
        (("elasticnet", 2.0, 0.2), np.float32),
    ],
)
def test_standardization_sparse(
    fit_intercept, reg_dtype, client, shift_scale=False
):
    regularization = reg_dtype[0]
    datatype = reg_dtype[1]

    n_rows = 10000
    n_cols = 25
    n_info = 15
    n_classes = 4
    nnz = int(n_rows * n_cols * 0.3)  # number of non-zero values
    tolerance = 0.005

    shift = (
        0.0
        if shift_scale is False
        else [random.uniform(-n_cols, n_cols) for _ in range(n_cols)]
    )
    scale = (
        1.0
        if shift_scale is False
        else [random.uniform(1.0, 10 * n_cols) for _ in range(n_cols)]
    )

    n_parts = 10
    max_iter = 5  # cannot set this too large. Observed GPU-specific coefficients when objective converges at 0.

    penalty = regularization[0]
    C = regularization[1]
    l1_ratio = regularization[2]

    est_params = {
        "penalty": penalty,
        "C": C,
        "l1_ratio": l1_ratio,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
    }

    def make_classification_with_nnz(
        datatype, n_rows, n_cols, n_info, n_classes, shift, scale, nnz
    ):
        assert n_rows * n_cols >= nnz

        X, y = make_classification_dataset(
            datatype,
            n_rows,
            n_cols,
            n_info,
            n_classes=n_classes,
            shift=shift,
            scale=scale,
        )
        X = X.flatten()
        num_zero = len(X) - nnz
        zero_indices = np.random.choice(
            a=range(len(X)), size=num_zero, replace=False
        )
        X[zero_indices] = 0
        X_res = X.reshape(n_rows, n_cols)
        return X_res, y

    X_origin, y = make_classification_with_nnz(
        datatype, n_rows, n_cols, n_info, n_classes, shift, scale, nnz
    )
    X = csr_matrix(X_origin)
    assert X.nnz == nnz and X.shape == (n_rows, n_cols)

    X_scaled, _, scaler = standardize_dataset(
        X_origin, X_origin, fit_intercept
    )

    X_da, y_da = _prep_training_data_sparse(
        client, X, y, partitions_per_worker=n_parts
    )
    from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

    assert X_da.shape == (n_rows, n_cols)

    computed_csr = X_da.compute()
    assert isinstance(computed_csr, csr_matrix)
    assert computed_csr.nnz == nnz and computed_csr.shape == (n_rows, n_cols)
    assert array_equal(computed_csr.data, X.data, unit_tol=tolerance)
    assert array_equal(computed_csr.indices, X.indices, unit_tol=tolerance)
    assert array_equal(computed_csr.indptr, X.indptr, unit_tol=tolerance)

    lr_on = cumlLBFGS_dask(standardization=True, verbose=True, **est_params)
    lr_on.fit(X_da, y_da)

    (
        lron_coef_origin,
        lron_intercept_origin,
    ) = adjust_standardization_model_for_comparison(
        lr_on.coef_, lr_on.intercept_, fit_intercept, scaler
    )

    from cuml.linear_model import LogisticRegression as SG

    sg = SG(**est_params)
    sg.fit(X_scaled, y)

    assert array_equal(lron_coef_origin, sg.coef_, unit_tol=tolerance)
    assert array_equal(
        lron_intercept_origin, sg.intercept_, unit_tol=tolerance
    )

    assert lr_on.dtype == datatype


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "reg_dtype",
    [
        ((None, 1.0, None), np.float64),
        (("l2", 2.0, None), np.float32),
        (("l1", 2.0, None), np.float64),
        (("elasticnet", 2.0, 0.2), np.float32),
    ],
)
def test_standardization_sparse_with_shift_scale(
    fit_intercept, reg_dtype, client
):
    test_standardization_sparse(
        fit_intercept, reg_dtype, client, shift_scale=True
    )


@pytest.mark.parametrize("standardization", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in divide:RuntimeWarning:sklearn"
)
def test_sparse_all_zeroes(
    standardization, fit_intercept, client, X=None, y=None, n_parts=2
):
    if X is None:
        X = np.array([(0, 0), (0, 0), (0, 0), (0, 0)], "float32")

    if y is None:
        y = np.array([1.0, 1.0, 0.0, 0.0], "float32")

    unit_tol = 0.001

    X_csr = csr_matrix(X)
    X_da_csr, y_da = _prep_training_data_sparse(client, X_csr, y, n_parts)

    from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

    mg = cumlLBFGS_dask(
        fit_intercept=fit_intercept,
        verbose=True,
        standardization=standardization,
    )
    mg.fit(X_da_csr, y_da)
    mg_preds = mg.predict(X_da_csr).compute()

    from sklearn.linear_model import LogisticRegression

    if standardization is False:
        X_cpu = X
    else:
        (
            X_cpu,
            _,
            scaler,
        ) = standardize_dataset(X, X, fit_intercept)

    cpu_lr = LogisticRegression(fit_intercept=fit_intercept)
    with ignore_deprecated_lbfgs_params_warning():
        cpu_lr.fit(X_cpu, y)
    cpu_preds = cpu_lr.predict(X_cpu)

    assert array_equal(mg_preds, cpu_preds)

    if standardization is False:
        mg_coef = mg.coef_
        mg_intercept = mg.intercept_
    else:
        mg_coef, mg_intercept = adjust_standardization_model_for_comparison(
            mg.coef_, mg.intercept_, fit_intercept, scaler
        )

    assert array_equal(
        mg_coef,
        cpu_lr.coef_,
        unit_tol=unit_tol,
        with_sign=True,
    )
    assert array_equal(
        mg_intercept,
        cpu_lr.intercept_,
        unit_tol=unit_tol,
        with_sign=True,
    )


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_sparse_one_gpu_all_zeroes(fit_intercept, client):
    """
    This test case requires two GPUs to function properly.
    """
    datatype = "float32"
    X = np.array([(10, 20), (0, 0), (0, 0), (0, 0)], datatype)
    y = np.array([1.0, 1.0, 0.0, 0.0], datatype)
    test_sparse_all_zeroes(
        standardization=True,
        fit_intercept=fit_intercept,
        client=client,
        X=X,
        y=y,
        n_parts=2,
    )
