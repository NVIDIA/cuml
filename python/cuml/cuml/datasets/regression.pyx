#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from random import randint

import cupy as cp

import cuml.internals.nvtx as nvtx
from cuml.internals import get_handle, mlfunc

from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/datasets/make_regression.hpp" namespace "ML" nogil:
    void cpp_make_regression "ML::Datasets::make_regression" (
        const handle_t& handle,
        float* out,
        float* values,
        long n_rows,
        long n_cols,
        long n_informative,
        float* coef,
        long n_targets,
        float bias,
        long effective_rank,
        float tail_strength,
        float noise,
        bool shuffle,
        uint64_t seed) except +

    void cpp_make_regression "ML::Datasets::make_regression" (
        const handle_t& handle,
        double* out,
        double* values,
        long n_rows,
        long n_cols,
        long n_informative,
        double* coef,
        long n_targets,
        double bias,
        long effective_rank,
        double tail_strength,
        double noise,
        bool shuffle,
        uint64_t seed) except +


@nvtx.annotate(message="datasets.make_regression", domain="cuml_python")
@mlfunc(array_arg=None)
def make_regression(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_targets=1,
    bias=0.0,
    effective_rank=None,
    tail_strength=0.5,
    noise=0.0,
    shuffle=True,
    coef=False,
    random_state=None,
    dtype='float32',
):
    """Generate a random regression problem.

    See https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

    Examples
    --------

    .. code-block:: python

        >>> from cuml.datasets.regression import make_regression
        >>> from cuml.linear_model import LinearRegression

        >>> # Create regression problem
        >>> data, values = make_regression(n_samples=200, n_features=12,
        ...                                n_informative=7, bias=-4.2,
        ...                                noise=0.3, random_state=10)

        >>> # Perform a linear regression on this problem
        >>> lr = LinearRegression()
        >>> reg = lr.fit(data, values)
        >>> print(reg.coef_) # doctest: +SKIP
        [-2.6980877e-02  7.7027252e+01  1.1498465e+01  8.5468025e+00
        5.8548538e+01  6.0772545e+01  3.6876743e+01  4.0023815e+01
        4.3908358e-03 -2.0275116e-02  3.5066366e-02 -3.4512520e-02]

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.
    n_features : int, optional (default=2)
        The number of features.
    n_informative : int, optional (default=2)
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.
    n_targets : int, optional (default=1)
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample. By default, the output is a scalar.
    bias : float, optional (default=0.0)
        The bias term in the underlying linear model.
    effective_rank : int or None, optional (default=None)
        if not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.
        if None:
            The input set is well conditioned, centered and gaussian with
            unit variance.
    tail_strength : float between 0.0 and 1.0, optional (default=0.5)
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None.
    noise : float, optional (default=0.0)
        The standard deviation of the gaussian noise applied to the output.
    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.
    coef : boolean, optional (default=False)
        If True, the coefficients of the underlying linear model are returned.
    random_state : int, RandomState instance or None (default)
        Seed for the random number generator for dataset creation.
    dtype: string or numpy dtype (default: 'float32')
        The output dtype. Only 'float32' or 'float64' supported.

    Returns
    -------
    X : array of shape (n_samples, n_features)
        The input samples.

    y : array of shape (n_samples,) or (n_samples, n_targets)
        The output values.

    coef : array of shape (n_features,) or (n_features, n_targets), optional
        The coefficient of the underlying linear model. It is returned only if
        coef is True.
    """  # noqa: E501
    dtype = cp.dtype(dtype)
    if dtype not in ["float32", "float64"]:
        raise ValueError(f"Expected dtype in ['float32', 'float64'], got `{dtype!s}`")

    if effective_rank is None:
        effective_rank = -1

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    X = cp.zeros((n_samples, n_features), dtype=dtype, order='C')
    cdef uintptr_t out_ptr = X.data.ptr

    y = cp.zeros((n_samples, n_targets), dtype=dtype, order='C')
    cdef uintptr_t values_ptr = y.data.ptr

    cdef uintptr_t coef_ptr
    coef_ptr = <uintptr_t> NULL
    if coef:
        coefs = cp.zeros((n_features, n_targets), dtype=dtype, order='C')
        coef_ptr = coefs.data.ptr

    if random_state is None:
        random_state = randint(0, 10**18)

    if dtype == "float32":
        cpp_make_regression(handle_[0], <float*> out_ptr,
                            <float*> values_ptr, <long> n_samples,
                            <long> n_features, <long> n_informative,
                            <float*> coef_ptr, <long> n_targets, <float> bias,
                            <long> effective_rank, <float> tail_strength,
                            <float> noise, <bool> shuffle,
                            <uint64_t> random_state)

    else:
        cpp_make_regression(handle_[0], <double*> out_ptr,
                            <double*> values_ptr, <long> n_samples,
                            <long> n_features, <long> n_informative,
                            <double*> coef_ptr, <long> n_targets,
                            <double> bias, <long> effective_rank,
                            <double> tail_strength, <double> noise,
                            <bool> shuffle, <uint64_t> random_state)

    if n_targets == 1:
        y = y.ravel()
        if coef:
            coefs = coefs.ravel()

    if coef:
        return X, y, coefs
    else:
        return X, y
