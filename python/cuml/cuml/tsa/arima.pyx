#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from typing import Mapping, Optional, Tuple, Union

import cupy as cp
import numpy as np

from cuml.internals import ReflectedAttr, logger, mlfunc, nvtx
from cuml.internals.base import Base, get_handle
from cuml.internals.validation import check_array
from cuml.tsa._deprecation import warn_deprecated_tsa_api
from cuml.tsa.batched_lbfgs import batched_fmin_lbfgs_b

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/tsa/arima_common.h" namespace "ML" nogil:
    cdef cppclass ARIMAParams[DataT]:
        DataT* mu
        DataT* beta
        DataT* ar
        DataT* ma
        DataT* sar
        DataT* sma
        DataT* sigma2

    cdef cppclass ARIMAMemory[DataT]:
        ARIMAMemory(const ARIMAOrder& order, int batch_size, int n_obs,
                    char* in_buf) except +

        @staticmethod
        size_t compute_size(const ARIMAOrder& order, int batch_size, int n_obs) except +


cdef extern from "cuml/tsa/batched_arima.hpp" namespace "ML" nogil:
    ctypedef enum LoglikeMethod: CSS, MLE

    void cpp_pack "pack" (
        handle_t& handle, const ARIMAParams[double]& params,
        const ARIMAOrder& order, int batch_size, double* param_vec) except +

    void cpp_unpack "unpack" (
        handle_t& handle, ARIMAParams[double]& params,
        const ARIMAOrder& order, int batch_size, const double* param_vec) except +

    bool detect_missing(
        handle_t& handle, const double* d_y, int n_elem) except +

    void batched_diff(
        handle_t& handle, double* d_y_diff, const double* d_y, int batch_size,
        int n_obs, const ARIMAOrder& order) except +

    void batched_loglike(
        handle_t& handle, const ARIMAMemory[double]& arima_mem,
        const double* y, const double* d_exog, int batch_size, int nobs,
        const ARIMAOrder& order, const double* params, double* loglike,
        bool trans, bool host_loglike, LoglikeMethod method, int truncate) except +

    void batched_loglike(
        handle_t& handle, const ARIMAMemory[double]& arima_mem,
        const double* y, const double* d_exog, int batch_size, int n_obs,
        const ARIMAOrder& order, const ARIMAParams[double]& params,
        double* loglike, bool trans, bool host_loglike, LoglikeMethod method,
        int truncate) except +

    void batched_loglike_grad(
        handle_t& handle, const ARIMAMemory[double]& arima_mem,
        const double* d_y, const double* d_exog, int batch_size, int nobs,
        const ARIMAOrder& order, const double* d_x, double* d_grad, double h,
        bool trans, LoglikeMethod method, int truncate) except +

    void cpp_predict "predict" (
        handle_t& handle, const ARIMAMemory[double]& arima_mem,
        const double* d_y, const double* d_exog, const double* d_exog_fut,
        int batch_size, int nobs, int start, int end, const ARIMAOrder& order,
        const ARIMAParams[double]& params, double* d_y_p, bool pre_diff,
        double level, double* d_lower, double* d_upper) except +

    void information_criterion(
        handle_t& handle, const ARIMAMemory[double]& arima_mem,
        const double* d_y, const double* d_exog, int batch_size, int nobs,
        const ARIMAOrder& order, const ARIMAParams[double]& params,
        double* ic, int ic_type) except +

    void estimate_x0(
        handle_t& handle, ARIMAParams[double]& params, const double* d_y,
        const double* d_exog, int batch_size, int nobs,
        const ARIMAOrder& order, bool missing) except +


cdef extern from "cuml/tsa/batched_kalman.hpp" namespace "ML" nogil:

    void batched_jones_transform(
        handle_t& handle, ARIMAMemory[double]& arima_mem,
        const ARIMAOrder& order, int batchSize, bool isInv,
        const double* h_params, double* h_Tparams) except +


cdef class ARIMAParamsWrapper:
    """A wrapper class for ARIMAParams"""
    cdef ARIMAParams[double] params

    def __cinit__(self, model):
        cdef ARIMAOrder order = model.order

        cdef uintptr_t d_mu_ptr = \
            model.mu_.data.ptr if order.k else <uintptr_t> NULL
        cdef uintptr_t d_beta_ptr = \
            model.beta_.data.ptr if order.n_exog else <uintptr_t> NULL
        cdef uintptr_t d_ar_ptr = \
            model.ar_.data.ptr if order.p else <uintptr_t> NULL
        cdef uintptr_t d_ma_ptr = \
            model.ma_.data.ptr if order.q else <uintptr_t> NULL
        cdef uintptr_t d_sar_ptr = \
            model.sar_.data.ptr if order.P else <uintptr_t> NULL
        cdef uintptr_t d_sma_ptr = \
            model.sma_.data.ptr if order.Q else <uintptr_t> NULL
        cdef uintptr_t d_sigma2_ptr = <uintptr_t> model.sigma2_.data.ptr

        self.params.mu = <double*> d_mu_ptr
        self.params.beta = <double*> d_beta_ptr
        self.params.ar = <double*> d_ar_ptr
        self.params.ma = <double*> d_ma_ptr
        self.params.sar = <double*> d_sar_ptr
        self.params.sma = <double*> d_sma_ptr
        self.params.sigma2 = <double*> d_sigma2_ptr


class ARIMA(Base):
    """
    Implements a batched ARIMA model for in- and out-of-sample
    time-series prediction, with support for seasonality (SARIMA)

    .. deprecated:: 26.08
        ``cuml.tsa.ARIMA`` and ``cuml.ARIMA`` are deprecated and will be
        removed in the cuML 26.12 release.

    ARIMA stands for Auto-Regressive Integrated Moving Average.
    See https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

    This class can fit an ARIMA(p,d,q) or ARIMA(p,d,q)(P,D,Q)_s model to a
    batch of time series of the same length (or various lengths, using missing
    values at the start for padding).
    The implementation is designed to give the best performance when using
    large batches of time series.

    Parameters
    ----------
    endog : dataframe or array-like (device or host)
        Endogenous variable, assumed to have each time series in columns.
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
        Missing values are accepted, represented by NaN.
    order : Tuple[int, int, int] (default=(1,1,1))
        The ARIMA order (p, d, q) of the model
    seasonal_order : Tuple[int, int, int, int] (default=(0,0,0,0))
        The seasonal ARIMA order (P, D, Q, s) of the model
    exog : dataframe or array-like (device or host) (default=None)
        Exogenous variables, assumed to have each time series in columns,
        such that variables associated with a same batch member are adjacent
        (number of columns: n_exog * batch_size)
        Acceptable formats: cuDF DataFrame, cuDF Series, NumPy ndarray,
        Numba device ndarray, cuda array interface compliant array like CuPy.
        Missing values are not supported.
    fit_intercept : bool or int (default = True)
        Whether to include a constant trend mu in the model
    simple_differencing : bool or int (default = True)
        If True, the data is differenced before being passed to the Kalman
        filter. If False, differencing is part of the state-space model.
        In some cases this setting can be ignored: computing forecasts with
        confidence intervals will force it to False ; fitting with the CSS
        method will force it to True.
        Note: that forecasts are always for the original series, whereas
        statsmodels computes forecasts for the differenced series when
        simple_differencing is True.
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
    order : ARIMAOrder
        The ARIMA order of the model (p, d, q, P, D, Q, s, k, n_exog)
    d_y : device array
        Time series data on device
    n_obs : int
        Number of observations
    batch_size : int
        Number of time series in the batch
    dtype : numpy.dtype
        Floating-point type of the data and parameters
    niter : numpy.ndarray
        After fitting, contains the number of iterations before convergence
        for each time series.

    Notes
    -----
    *Performance:* Let :math:`r=max(p+s*P, q+s*Q+1)`. The device memory used
    for most operations is
    :math:\
    `O(\\mathtt{batch\\_size}*\\mathtt{n\\_obs} + \\mathtt{batch\\_size}*r^2)`.
    The execution time is a linear function of `n_obs` and `batch_size`
    (if `batch_size` is large), but grows very fast with `r`.

    The performance is optimized for very large batch sizes (e.g thousands of
    series).

    References
    ----------
    This class is heavily influenced by the Python library `statsmodels`,
    particularly `statsmodels.tsa.statespace.sarimax.SARIMAX`.
    See https://www.statsmodels.org/stable/statespace.html.

    Additionally the following book is a useful reference:
    "Time Series Analysis by State Space Methods",
    J. Durbin, S.J. Koopman, 2nd Edition (2012).

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.tsa.arima import ARIMA

        >>> # Create seasonal data with a trend, a seasonal pattern and noise
        >>> n_obs = 100
        >>> cp.random.seed(12)
        >>> x = cp.linspace(0, 1, n_obs)
        >>> pattern = cp.array([[0.05, 0.0], [0.07, 0.03],
        ...                     [-0.03, 0.05], [0.02, 0.025]])
        >>> noise = cp.random.normal(scale=0.01, size=(n_obs, 2))
        >>> y = (cp.column_stack((0.5*x, -0.25*x)) + noise
        ...     + cp.tile(pattern, (25, 1)))

        >>> # Fit a seasonal ARIMA model
        >>> model = ARIMA(y,
        ...               order=(0,1,1),
        ...               seasonal_order=(0,1,1,4),
        ...               fit_intercept=False)
        >>> model.fit()
        ARIMA(...)
        >>> # Forecast
        >>> fc = model.forecast(10)
        >>> print(fc) # doctest: +SKIP
        [[ 0.55204599 -0.25681163]
        [ 0.57430705 -0.2262438 ]
        [ 0.48120315 -0.20583011]
        [ 0.535594   -0.24060046]
        [ 0.57207541 -0.26695497]
        [ 0.59433647 -0.23638713]
        [ 0.50123257 -0.21597344]
        [ 0.55562342 -0.25074379]
        [ 0.59210483 -0.27709831]
        [ 0.61436589 -0.24653047]]

    """

    d_y = ReflectedAttr()
    # TODO: (MDD) Should this be public? Its not listed in the attributes doc
    _d_y_diff = ReflectedAttr()
    _temp_mem = ReflectedAttr()

    mu_ = ReflectedAttr()
    beta_ = ReflectedAttr()
    ar_ = ReflectedAttr()
    ma_ = ReflectedAttr()
    sar_ = ReflectedAttr()
    sma_ = ReflectedAttr()
    sigma2_ = ReflectedAttr()

    def __init__(self,
                 endog,
                 *,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 exog=None,
                 fit_intercept=True,
                 simple_differencing=True,
                 verbose=False,
                 output_type=None):

        warn_deprecated_tsa_api("cuml.tsa.ARIMA")

        # Initialize base class
        super().__init__(verbose=verbose, output_type=output_type)
        self._set_output_type(endog)

        # Check validity of the ARIMA order and seasonal order
        p, d, q = order
        P, D, Q, s = seasonal_order
        if P + D + Q > 0 and s < 2:
            raise ValueError("ERROR: Invalid period for seasonal ARIMA: {}"
                             .format(s))
        if d + D > 2:
            raise ValueError("ERROR: Invalid order. Required: d+D <= 2")
        if s != 0 and (p >= s or q >= s):
            raise ValueError("ERROR: Invalid order. Required: s > p, s > q")
        if p + q + P + Q + int(fit_intercept) == 0:
            raise ValueError("ERROR: Invalid order. At least one parameter"
                             " among p, q, P, Q and fit_intercept must be"
                             " non-zero")
        if p > 8 or P > 8 or q > 8 or Q > 8:
            raise ValueError("ERROR: Invalid order. Required: p,q,P,Q <= 8")
        if max(p + s * P, q + s * Q) > 1024:
            raise ValueError("ERROR: Invalid order. "
                             "Required: max(p+s*P, q+s*Q) <= 1024")

        # Endogenous variable. Float64 only for now.
        self.d_y = d_y = check_array(
            endog,
            dtype="float64",
            order="F",
            ensure_2d=False,
            ensure_all_finite=False,
        )
        self.n_obs = d_y.shape[0]
        self.batch_size = d_y.shape[1] if d_y.ndim == 2 else 1
        self.dtype = d_y.dtype

        if self.n_obs < d + s * D + 1:
            raise ValueError("ERROR: Number of observations too small for the"
                             " given order")

        # Exogenous variables
        if exog is not None:
            self.d_exog = d_exog = check_array(
                exog,
                dtype="float64",
                order="F",
                ensure_2d=False,
                ensure_all_finite=False,
            )
            n_obs_exog = d_exog.shape[0]
            n_cols_exog = d_exog.shape[1] if d_exog.ndim == 2 else 1

            if n_cols_exog % self.batch_size != 0:
                raise ValueError("Number of columns in exog is not a multiple"
                                 " of batch_size")
            if n_obs_exog != self.n_obs:
                raise ValueError("Number of observations mismatch between"
                                 " endog and exog")

            n_exog = n_cols_exog // self.batch_size
        else:
            n_exog = 0

        # Set the ARIMA order
        cdef ARIMAOrder cpp_order
        cpp_order.p, cpp_order.d, cpp_order.q = order
        cpp_order.P, cpp_order.D, cpp_order.Q, cpp_order.s = seasonal_order
        cpp_order.k = int(fit_intercept)
        cpp_order.n_exog = n_exog
        self.order = cpp_order

        self.simple_differencing = simple_differencing

        self._d_y_diff = cp.empty(
            (self.n_obs - d - s * D, self.batch_size),
            dtype=self.dtype,
            order="F",
        )
        if n_exog > 0:
            self._d_exog_diff = cp.empty(
                (self.n_obs - d - s * D, self.batch_size * n_exog),
                dtype=self.dtype,
                order="F",
            )

        self.n_obs_diff = self.n_obs - d - D * s

        # Allocate temporary storage
        temp_mem_size = ARIMAMemory[double].compute_size(
            cpp_order, <int> self.batch_size, <int> self.n_obs)
        self._temp_mem = cp.empty(temp_mem_size, dtype="int8")

        self._initial_calc()

    @mlfunc(convert_output=False)
    def _initial_calc(self):
        """
        This separates the initial calculation from the initialization to make
        ReflectedAttr work.
        """

        cdef uintptr_t d_y_ptr = self.d_y.data.ptr
        cdef uintptr_t d_y_diff_ptr = self._d_y_diff.data.ptr
        cdef uintptr_t d_exog_ptr
        cdef uintptr_t d_exog_diff_ptr
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef ARIMAOrder cpp_order_diff = self.order

        # Detect missing observations
        self.missing = detect_missing(handle_[0], <double*> d_y_ptr,
                                      <int> self.batch_size * self.n_obs)
        if self.missing and self.simple_differencing:
            logger.warn("Missing observations detected."
                        " Forcing simple_differencing=False")
            self.simple_differencing = False

        if self.simple_differencing:
            # Compute the differenced series
            batched_diff(handle_[0], <double*> d_y_diff_ptr, <double*> d_y_ptr,
                         <int> self.batch_size, <int> self.n_obs, self.order)

            # Create a version of the order for the differenced series
            cpp_order_diff.d = 0
            cpp_order_diff.D = 0
            self.order_diff = cpp_order_diff

            if cpp_order_diff.n_exog > 0:
                d_exog_ptr = self.d_exog.data.ptr
                d_exog_diff_ptr = self._d_exog_diff.data.ptr
                batched_diff(handle_[0], <double*> d_exog_diff_ptr,
                             <double*> d_exog_ptr,
                             <int> self.batch_size * cpp_order_diff.n_exog,
                             <int> self.n_obs, self.order)
        else:
            self.order_diff = None

    def __str__(self):
        cdef ARIMAOrder order = self.order
        intercept_str = 'c' if order.k else 'n'
        if order.s:
            return "ARIMA({},{},{})({},{},{})_{} ({}) - {} series".format(
                order.p, order.d, order.q, order.P, order.D, order.Q, order.s,
                intercept_str, self.batch_size)
        else:
            return "ARIMA({},{},{}) ({}) - {} series".format(
                order.p, order.d, order.q, intercept_str, self.batch_size)

    def __repr__(self):
        # ARIMA can't use the base `__repr__` because it does not support
        # `get_params`.
        cdef ARIMAOrder order = self.order
        return (
            "ARIMA(order={}, seasonal_order={}, fit_intercept={}, "
            "simple_differencing={})".format(
                (order.p, order.d, order.q),
                (order.P, order.D, order.Q, order.s),
                order.k != 0,
                self.simple_differencing,
            )
        )

    @nvtx.annotate(message="tsa.arima.ARIMA._ic", domain="cuml_python")
    def _ic(self, ic_type: str):
        """Wrapper around C++ information_criterion
        """
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        cdef ARIMAOrder order = self.order
        cdef ARIMAOrder order_kf = \
            self.order_diff if self.simple_differencing else self.order
        cdef ARIMAParams[double] cpp_params = ARIMAParamsWrapper(self).params

        ic = cp.empty(self.batch_size, self.dtype)
        cdef uintptr_t d_ic_ptr = ic.data.ptr
        cdef uintptr_t d_y_kf_ptr = \
            self._d_y_diff.data.ptr if self.simple_differencing else self.d_y.data.ptr

        cdef uintptr_t d_exog_kf_ptr = <uintptr_t> NULL
        if order.n_exog:
            d_exog_kf_ptr = (self._d_exog_diff.data.ptr if self.simple_differencing
                             else self.d_exog.data.ptr)

        n_obs_kf = (self.n_obs_diff if self.simple_differencing
                    else self.n_obs)

        ic_name_to_number = {"aic": 0, "aicc": 1, "bic": 2}
        cdef int ic_type_id
        try:
            ic_type_id = ic_name_to_number[ic_type.lower()]
        except KeyError as e:
            raise NotImplementedError("IC type '{}' unknown".format(ic_type))

        cdef uintptr_t d_temp_mem = self._temp_mem.data.ptr
        arima_mem_ptr = new ARIMAMemory[double](
            order, <int> self.batch_size, <int> self.n_obs,
            <char*> d_temp_mem)

        information_criterion(handle_[0], arima_mem_ptr[0],
                              <double*> d_y_kf_ptr, <double*> d_exog_kf_ptr,
                              <int> self.batch_size, <int> n_obs_kf, order_kf,
                              cpp_params, <double*> d_ic_ptr,
                              <int> ic_type_id)

        del arima_mem_ptr

        return ic

    @property
    @mlfunc
    def aic(self):
        """Akaike Information Criterion"""
        return self._ic("aic")

    @property
    @mlfunc
    def aicc(self):
        """Corrected Akaike Information Criterion"""
        return self._ic("aicc")

    @property
    @mlfunc
    def bic(self):
        """Bayesian Information Criterion"""
        return self._ic("bic")

    @property
    def complexity(self):
        """Model complexity (number of parameters)"""
        cdef ARIMAOrder order = self.order
        return (order.p + order.P + order.q + order.Q + order.k + order.n_exog
                + 1)

    @mlfunc
    def get_fit_params(self):
        """Get all the fit parameters. Not to be confused with get_params
        Note: pack() can be used to get a compact vector of the parameters

        Returns
        -------
        params: Dict[str, array-like]
            A dictionary of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu and sigma2 and
            (n, batch_size) for any other type, where n is the corresponding
            number of parameters of this type.
        """
        cdef ARIMAOrder order = self.order
        params = dict()
        names = ["mu", "beta", "ar", "ma", "sar", "sma", "sigma2"]
        criteria = [order.k, order.n_exog, order.p, order.q, order.P, order.Q,
                    True]
        for i in range(len(names)):
            if criteria[i] > 0:
                params[names[i]] = getattr(self, "{}_".format(names[i]))
        return params

    def set_fit_params(self, params: Mapping[str, object]):
        """Set all the fit parameters. Not to be confused with ``set_params``
        Note: `unpack()` can be used to load a compact vector of the
        parameters

        Parameters
        ----------
        params: Mapping[str, array-like]
            A dictionary of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu and sigma2 and
            (n, batch_size) for any other type, where n is the corresponding
            number of parameters of this type.
        """
        for param_name in ["mu", "beta", "ar", "ma", "sar", "sma", "sigma2"]:
            if param_name in params:
                array = check_array(
                    params[param_name],
                    dtype="float64",
                    ensure_2d=False,
                    ensure_all_finite=False,
                )
                setattr(self, "{}_".format(param_name), array)

    @classmethod
    def _get_param_names(cls):
        """
        ARIMA is unable to be cloned at this time.
        The methods: `_get_param_names()`, `get_params` and
        set_params` will raise ``NotImplementedError``
        """
        raise NotImplementedError("ARIMA is unable to be cloned via "
                                  "`get_params` and `set_params`.")

    def get_params(self, deep=True):
        """
        ARIMA is unable to be cloned at this time.
        The methods: `_get_param_names()`, `get_params` and
        `set_params` will raise ``NotImplementedError``
        """
        raise NotImplementedError("ARIMA is unable to be cloned via "
                                  "`get_params` and `set_params`.")

    def set_params(self, **params):
        """
        ARIMA is unable to be cloned at this time.
        The methods: `_get_param_names()`, `get_params` and
        `set_params` will raise ``NotImplementedError``
        """
        raise NotImplementedError("ARIMA is unable to be cloned via "
                                  "`get_params` and `set_params`.")

    @mlfunc(array_arg=None)
    def predict(
        self,
        start=0,
        end=None,
        level=None,
        exog=None,
    ):
        """Compute in-sample and/or out-of-sample prediction for each series

        Parameters
        ----------
        start : int (default = 0)
            Index where to start the predictions (0 <= start <= num_samples)
        end : int (default = None)
            Index where to end the predictions, excluded (end > start), or
            ``None`` to predict until the last observation
        level : float or None (default = None)
            Confidence level for prediction intervals, or None to return only
            the point forecasts. ``0 < level < 1``
        exog : dataframe or array-like (device or host)
            Future values for exogenous variables. Assumed to have each time
            series in columns, such that variables associated with a same
            batch member are adjacent.
            Shape = (end - n_obs, n_exog * batch_size)

        Returns
        -------
        y_p : array-like (device)
            Predictions. Shape = (end - start, batch_size)
        lower: array-like (device) (optional)
            Lower limit of the prediction interval if ``level != None``
            Shape = (end - start, batch_size)
        upper: array-like (device) (optional)
            Upper limit of the prediction interval if ``level != None``
            Shape = (end - start, batch_size)

        Examples
        --------
        .. code-block:: python

            from cuml.tsa.arima import ARIMA

            model = ARIMA(ys, order=(1,1,1))
            model.fit()
            y_pred = model.predict()
        """
        cdef ARIMAOrder order = self.order
        cdef ARIMAParams[double] cpp_params = ARIMAParamsWrapper(self).params

        if start < 0:
            raise ValueError("ERROR(`predict`): start < 0")
        elif start > self.n_obs:
            raise ValueError("ERROR(`predict`): There can't be a gap between"
                             " the data and the prediction")
        elif end <= start:
            raise ValueError("ERROR(`predict`): end <= start")
        elif self.simple_differencing and start < order.d + order.D * order.s:
            logger.warn("Predictions before {} are undefined when using"
                        " simple_differencing=True, will be set to NaN"
                        .format(order.d + order.D * order.s))

        if level is not None:
            if level <= 0 or level >= 1:
                raise ValueError("ERROR: Invalid confidence level: {}"
                                 .format(level))
            elif level > 0 and start < self.n_obs:
                raise ValueError("ERROR: Prediction intervals can only be"
                                 " computed for out-of-sample predictions")

        if end is None:
            end = self.n_obs

        if order.n_exog > 0 and end > self.n_obs and exog is None:
            raise ValueError("The model was fit with a regression component,"
                             " so future values must be provided via `exog`")
        elif order.n_exog == 0 and exog is not None:
            raise ValueError("A value was given for `exog` but the model was"
                             " fit without any regression component")
        elif end <= self.n_obs and exog is not None:
            raise ValueError("A value was given for `exog` but only in-sample"
                             " predictions were requested")

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        predict_size = end - start

        # Future values of the exogenous variables
        cdef uintptr_t d_exog_fut_ptr = <uintptr_t> NULL
        if order.n_exog and end > self.n_obs:
            d_exog_fut = check_array(
                exog,
                dtype="float64",
                order="F",
                ensure_2d=False,
                ensure_all_finite=False,
            )
            n_obs_fut = d_exog_fut.shape[0]
            n_cols_fut = d_exog_fut.shape[1] if d_exog_fut.ndim == 2 else 1
            if n_obs_fut != end - self.n_obs:
                raise ValueError(
                    "Dimensions mismatch: `exog` should contain {}"
                    " observations per column".format(end - self.n_obs))
            elif n_cols_fut != self.batch_size * order.n_exog:
                raise ValueError(
                    "Dimensions mismatch: `exog` should have {} columns"
                    .format(self.batch_size * order.n_exog))
            d_exog_fut_ptr = d_exog_fut.data.ptr

        # allocate predictions and intervals device memory
        cdef uintptr_t d_y_p_ptr = <uintptr_t> NULL
        cdef uintptr_t d_lower_ptr = <uintptr_t> NULL
        cdef uintptr_t d_upper_ptr = <uintptr_t> NULL
        d_y_p = cp.empty((predict_size, self.batch_size), dtype="float64", order="F")
        d_y_p_ptr = d_y_p.data.ptr
        if level is not None:
            d_lower = cp.empty(
                (predict_size, self.batch_size), dtype="float64", order="F"
            )
            d_upper = cp.empty(
                (predict_size, self.batch_size), dtype="float64", order="F"
            )
            d_lower_ptr = d_lower.data.ptr
            d_upper_ptr = d_upper.data.ptr

        cdef uintptr_t d_y_ptr = self.d_y.data.ptr
        cdef uintptr_t d_exog_ptr = <uintptr_t> NULL
        if order.n_exog:
            d_exog_ptr = self.d_exog.data.ptr

        cdef uintptr_t d_temp_mem = self._temp_mem.data.ptr
        arima_mem_ptr = new ARIMAMemory[double](
            order, <int> self.batch_size, <int> self.n_obs,
            <char*> d_temp_mem)

        cpp_predict(handle_[0], arima_mem_ptr[0], <double*>d_y_ptr,
                    <double*>d_exog_ptr, <double*>d_exog_fut_ptr,
                    <int> self.batch_size, <int> self.n_obs, <int> start,
                    <int> end, order, cpp_params, <double*>d_y_p_ptr,
                    <bool> self.simple_differencing,
                    <double> (0 if level is None else level),
                    <double*> d_lower_ptr, <double*> d_upper_ptr)

        del arima_mem_ptr

        if level is None:
            return d_y_p
        else:
            return (d_y_p,
                    d_lower,
                    d_upper)

    @nvtx.annotate(message="tsa.arima.ARIMA.forecast", domain="cuml_python")
    @mlfunc(array_arg=None)
    def forecast(
        self,
        nsteps: int,
        level=None,
        exog=None
    ):
        """Forecast the given model `nsteps` into the future.

        Parameters
        ----------
        nsteps : int
            The number of steps to forecast beyond end of the given series
        level : float or None (default = None)
            Confidence level for prediction intervals, or None to return only
            the point forecasts. 0 < level < 1
        exog : dataframe or array-like (device or host) (default=None)
            Future values for exogenous variables. Assumed to have each time
            series in columns, such that variables associated with a same
            batch member are adjacent.
            Shape = (nsteps, n_exog * batch_size)

        Returns
        -------
        y_fc : array-like
            Forecasts. Shape = (nsteps, batch_size)
        lower : array-like (device) (optional)
            Lower limit of the prediction interval if level != None
            Shape = (end - start, batch_size)
        upper : array-like (device) (optional)
            Upper limit of the prediction interval if level != None
            Shape = (end - start, batch_size)

        Examples
        --------
        .. code-block:: python

            from cuml.tsa.arima import ARIMA
            ...
            model = ARIMA(ys, order=(1,1,1))
            model.fit()
            y_fc = model.forecast(10)
        """

        return self.predict(self.n_obs, self.n_obs + nsteps, level, exog)

    def _create_arrays(self):
        """Create the parameter arrays if non-existing"""
        cdef ARIMAOrder order = self.order

        if order.k and not hasattr(self, "mu_"):
            self.mu_ = cp.empty(self.batch_size, dtype="float64", order="F")
        if order.n_exog and not hasattr(self, "beta_"):
            self.beta_ = cp.empty(
                (order.n_exog, self.batch_size), dtype="float64", order="F"
            )
        if order.p and not hasattr(self, "ar_"):
            self.ar_ = cp.empty((order.p, self.batch_size), dtype="float64", order="F")
        if order.q and not hasattr(self, "ma_"):
            self.ma_ = cp.empty((order.q, self.batch_size), dtype="float64", order="F")
        if order.P and not hasattr(self, "sar_"):
            self.sar_ = cp.empty((order.P, self.batch_size), dtype="float64", order="F")
        if order.Q and not hasattr(self, "sma_"):
            self.sma_ = cp.empty((order.Q, self.batch_size), dtype="float64", order="F")
        if not hasattr(self, "sigma2_"):
            self.sigma2_ = cp.empty(self.batch_size, dtype="float64", order="F")

    @nvtx.annotate(message="tsa.arima.ARIMA._estimate_x0",
                   domain="cuml_python")
    @mlfunc(convert_output=False)
    def _estimate_x0(self):
        """Internal method. Estimate initial parameters of the model.
        """
        self._create_arrays()

        cdef ARIMAOrder order = self.order
        cdef ARIMAParams[double] cpp_params = ARIMAParamsWrapper(self).params

        cdef uintptr_t d_y_ptr = self.d_y.data.ptr
        cdef uintptr_t d_exog_ptr = <uintptr_t> NULL
        if order.n_exog:
            d_exog_ptr = self.d_exog.data.ptr
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        # Call C++ function
        estimate_x0(handle_[0], cpp_params, <double*> d_y_ptr,
                    <double*> d_exog_ptr, <int> self.batch_size,
                    <int> self.n_obs, order, <bool> self.missing)

    @mlfunc(convert_output=False)
    def fit(self,
            start_params: Optional[Mapping[str, object]] = None,
            opt_disp: int = -1,
            h: float = 1e-8,
            maxiter: int = 1000,
            method="ml",
            truncate: int = 0) -> "ARIMA":
        r"""Fit the ARIMA model to each time series.

        Parameters
        ----------
        start_params : Mapping[str, array-like] (optional)
            A mapping (e.g dictionary) of parameter names and associated arrays
            The key names are in {"mu", "ar", "ma", "sar", "sma", "sigma2"}
            The shape of the arrays are (batch_size,) for mu and sigma2
            parameters and (n, batch_size) for any other type, where n is the
            corresponding number of parameters of this type.
            Pass None for automatic estimation (recommended)

        opt_disp : int
            Fit diagnostic level (for L-BFGS solver):

            * `-1` for no output (default)
            * `0<n<100` for output every `n` steps
            * `n>100` for more detailed output

        h : float (default=1e-8)
            Finite-differencing step size. The gradient is computed using
            forward finite differencing:
            :math:`g = \frac{f(x + \mathtt{h}) - f(x)}{\mathtt{h}} + O(\mathtt{h})`

        maxiter : int (default=1000)
            Maximum number of iterations of L-BFGS-B
        method : str (default="ml")
            Estimation method - "css", "css-ml" or "ml".
            CSS uses a sum-of-squares approximation.
            ML estimates the log-likelihood with statespace methods.
            CSS-ML starts with CSS and refines with ML.
        truncate : int (default=0)
            When using CSS, start the sum of squares after a given number of
            observations
        """  # noqa
        def fit_helper(x_in, fit_method):

            def f(x: np.ndarray) -> np.ndarray:
                """The (batched) energy functional returning the negative
                log-likelihood (foreach series)."""
                # Recall: We maximize LL by minimizing -LL
                n_llf = -self._loglike(x, True, fit_method, truncate)
                return n_llf / (self.n_obs - 1)

            # Optimized finite differencing gradient for batches
            def gf(x) -> np.ndarray:
                """The gradient of the (batched) energy functional."""
                # Recall: We maximize LL by minimizing -LL
                n_gllf = -self._loglike_grad(x, h, True, fit_method, truncate)
                return n_gllf / (self.n_obs - 1)

            # Check initial parameter sanity
            if ((np.isnan(x_in).any()) or (np.isinf(x_in).any())):
                raise FloatingPointError(
                    "Initial parameter vector x has NaN or Inf.")

            # Optimize parameters by minimizing log likelihood.
            x_out, niter, flags = batched_fmin_lbfgs_b(
                f, x_in, self.batch_size, gf, iprint=opt_disp, factr=1000,
                maxiter=maxiter)

            # Handle non-zero flags with Warning
            if (flags != 0).any():
                logger.warn("fit: Some batch members had optimizer problems")

            return x_out, niter

        if start_params is None:
            self._estimate_x0()
        else:
            self.set_fit_params(start_params)

        x0 = self._batched_transform(self.pack(), True)

        method = method.lower()
        if method not in {"css", "css-ml", "ml"}:
            raise ValueError("Unknown method: {}".format(method))
        if self.missing and (method == "css" or method == "css-ml"):
            logger.warn("Missing observations detected."
                        " Forcing method=\"ml\"")
            method = "ml"
        if method == "css" or method == "css-ml":
            x, self.niter = fit_helper(x0, "css")
        if method == "css-ml" or method == "ml":
            x, niter = fit_helper(x if method == "css-ml" else x0, "ml")
            self.niter = (self.niter + niter) if method == "css-ml" else niter

        self.unpack(self._batched_transform(x))
        return self

    @nvtx.annotate(message="tsa.arima.ARIMA._loglike", domain="cuml_python")
    @mlfunc(convert_output=False)
    def _loglike(self, x, trans=True, method="ml", truncate=0):
        """Compute the batched log-likelihood for the given parameters.

        Parameters
        ----------
        x : array-like
            Packed parameter array, grouped by series
        trans : bool (default=True)
            Should the Jones' transform be applied?
            Note: The parameters from a fit model are already transformed.
        method : str (default="ml")
            Estimation method: "css" for sum-of-squares, "ml" for
            an estimation with statespace methods
        truncate : int (default=0)
            When using CSS, start the sum of squares after a given number of
            observations

        Returns
        -------
        loglike : numpy.ndarray
            Batched log-likelihood. Shape: (batch_size,)
        """
        cdef vector[double] vec_loglike
        vec_loglike.resize(self.batch_size)

        cdef LoglikeMethod ll_method = CSS if method == "css" else MLE
        diff = ll_method != MLE or self.simple_differencing

        cdef ARIMAOrder order = self.order
        cdef ARIMAOrder order_kf = self.order_diff if diff else self.order

        d_x_array = check_array(
            x,
            dtype="float64",
            order="C",
            ensure_2d=False,
            ensure_all_finite=False,
        )
        cdef uintptr_t d_x_ptr = d_x_array.data.ptr

        cdef uintptr_t d_y_kf_ptr = \
            self._d_y_diff.data.ptr if diff else self.d_y.data.ptr

        cdef uintptr_t d_exog_kf_ptr = <uintptr_t> NULL
        if order.n_exog:
            d_exog_kf_ptr = self._d_exog_diff.data.ptr if diff else self.d_exog.data.ptr

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        n_obs_kf = (self.n_obs_diff if diff else self.n_obs)

        cdef uintptr_t d_temp_mem = self._temp_mem.data.ptr
        arima_mem_ptr = new ARIMAMemory[double](
            order, <int> self.batch_size, <int> self.n_obs,
            <char*> d_temp_mem)

        batched_loglike(handle_[0], arima_mem_ptr[0], <double*> d_y_kf_ptr,
                        <double*> d_exog_kf_ptr, <int> self.batch_size,
                        <int> n_obs_kf, order_kf, <double*> d_x_ptr,
                        <double*> vec_loglike.data(), <bool> trans,
                        <bool> True, ll_method, <int> truncate)

        del arima_mem_ptr

        return np.array(vec_loglike, dtype=np.float64)

    @nvtx.annotate(message="tsa.arima.ARIMA._loglike_grad",
                   domain="cuml_python")
    @mlfunc(convert_output=False)
    def _loglike_grad(self, x, h=1e-8, trans=True, method="ml", truncate=0):
        """Compute the gradient (via finite differencing) of the batched
        log-likelihood.

        Parameters
        ----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        h : float
            The finite-difference stepsize
        trans : bool (default=True)
            Should the Jones' transform be applied?
            Note: The parameters from a fit model are already transformed.
        method : str (default="ml")
            Estimation method: "css" for sum-of-squares, "ml" for
            an estimation with statespace methods
        truncate : int (default=0)
            When using CSS, start the sum of squares after a given number of
            observations

        Returns
        -------
        grad : numpy.ndarray
            Batched log-likelihood gradient. Shape: (n_params * batch_size,)
            where n_params is the complexity of the model
        """
        N = self.complexity
        assert len(x) == N * self.batch_size

        cdef LoglikeMethod ll_method = CSS if method == "css" else MLE
        diff = ll_method != MLE or self.simple_differencing

        grad = cp.empty(N * self.batch_size, dtype="float64", order="F")
        cdef uintptr_t d_grad = <uintptr_t> grad.data.ptr

        cdef ARIMAOrder order = self.order
        cdef ARIMAOrder order_kf = self.order_diff if diff else self.order

        d_x_array = check_array(
            x,
            dtype="float64",
            order="C",
            ensure_2d=False,
            ensure_all_finite=False,
        )
        cdef uintptr_t d_x_ptr = d_x_array.data.ptr

        cdef uintptr_t d_y_kf_ptr = \
            self._d_y_diff.data.ptr if diff else self.d_y.data.ptr

        cdef uintptr_t d_exog_kf_ptr = <uintptr_t> NULL
        if order.n_exog:
            d_exog_kf_ptr = self._d_exog_diff.data.ptr if diff else self.d_exog.data.ptr

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        cdef uintptr_t d_temp_mem = self._temp_mem.data.ptr
        arima_mem_ptr = new ARIMAMemory[double](
            order, <int> self.batch_size, <int> self.n_obs,
            <char*> d_temp_mem)

        batched_loglike_grad(handle_[0], arima_mem_ptr[0],
                             <double*> d_y_kf_ptr, <double*> d_exog_kf_ptr,
                             <int> self.batch_size,
                             <int> (self.n_obs_diff if diff else self.n_obs),
                             order_kf, <double*> d_x_ptr, <double*> d_grad,
                             <double> h, <bool> trans, ll_method,
                             <int> truncate)

        del arima_mem_ptr

        return grad.get(order="A")

    @property
    @mlfunc(convert_output=False)
    def llf(self):
        """Log-likelihood of a fit model. Shape: (batch_size,)
        """
        # Implementation note: this is slightly different from batched_loglike
        # as it uses the device parameter arrays and not a host vector.
        # Also, it always uses the MLE method, trans=False and truncate=0

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        cdef vector[double] vec_loglike
        vec_loglike.resize(self.batch_size)

        cdef ARIMAOrder order = self.order
        cdef ARIMAOrder order_kf = \
            self.order_diff if self.simple_differencing else self.order
        cdef ARIMAParams[double] cpp_params = ARIMAParamsWrapper(self).params

        cdef uintptr_t d_y_kf_ptr = \
            self._d_y_diff.data.ptr if self.simple_differencing else self.d_y.data.ptr

        cdef uintptr_t d_exog_kf_ptr = <uintptr_t> NULL
        if order.n_exog:
            d_exog_kf_ptr = (self._d_exog_diff.data.ptr if self.simple_differencing
                             else self.d_exog.data.ptr)

        n_obs_kf = (self.n_obs_diff if self.simple_differencing
                    else self.n_obs)

        cdef LoglikeMethod ll_method = MLE

        cdef uintptr_t d_temp_mem = self._temp_mem.data.ptr
        arima_mem_ptr = new ARIMAMemory[double](
            order, <int> self.batch_size, <int> self.n_obs,
            <char*> d_temp_mem)

        batched_loglike(handle_[0], arima_mem_ptr[0], <double*> d_y_kf_ptr,
                        <double*> d_exog_kf_ptr, <int> self.batch_size,
                        <int> n_obs_kf, order_kf, cpp_params,
                        <double*> vec_loglike.data(), <bool> False,
                        <bool> True, ll_method, <int> 0)

        del arima_mem_ptr

        return np.array(vec_loglike, dtype=np.float64)

    @nvtx.annotate(message="tsa.arima.ARIMA.unpack", domain="cuml_python")
    @mlfunc(convert_output=False)
    def unpack(self, x: Union[list, np.ndarray]):
        """Unpack linearized parameter vector `x` into the separate
        parameter arrays of the model

        Parameters
        ----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        """
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        self._create_arrays()

        cdef ARIMAOrder order = self.order
        cdef ARIMAParams[double] cpp_params = ARIMAParamsWrapper(self).params

        d_x_array = check_array(
            x,
            dtype="float64",
            order="C",
            ensure_2d=False,
            ensure_all_finite=False,
        )
        cdef uintptr_t d_x_ptr = d_x_array.data.ptr

        cpp_unpack(handle_[0], cpp_params, order, <int> self.batch_size,
                   <double*>d_x_ptr)

    @nvtx.annotate(message="tsa.arima.ARIMA.pack", domain="cuml_python")
    @mlfunc(convert_output=False)
    def pack(self) -> np.ndarray:
        """Pack parameters of the model into a linearized vector `x`

        Returns
        -------
        x : numpy ndarray
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        """
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        cdef ARIMAOrder order = self.order
        cdef ARIMAParams[double] cpp_params = ARIMAParamsWrapper(self).params

        d_x_array = cp.empty(
            self.complexity * self.batch_size, dtype="float64", order="F"
        )
        cdef uintptr_t d_x_ptr = d_x_array.data.ptr

        cpp_pack(handle_[0], cpp_params, order, <int> self.batch_size,
                 <double*>d_x_ptr)

        return d_x_array.get(order="A")

    @nvtx.annotate(message="tsa.arima.ARIMA._batched_transform",
                   domain="cuml_python")
    def _batched_transform(self, x, isInv=False):
        """Applies Jones transform or inverse transform to a parameter vector

        Parameters
        ----------
        x : array-like
            Packed parameter array, grouped by series.
            Shape: (n_params * batch_size,)

        Returns
        -------
        Tx : array-like
            Packed transformed parameter array, grouped by series.
            Shape: (n_params * batch_size,)
        """
        cdef ARIMAOrder order = self.order
        N = self.complexity

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        Tx = np.zeros(self.batch_size * N)

        cdef uintptr_t d_temp_mem = self._temp_mem.data.ptr
        arima_mem_ptr = new ARIMAMemory[double](
            order, <int> self.batch_size, <int> self.n_obs,
            <char*> d_temp_mem)

        cdef uintptr_t x_ptr = x.ctypes.data
        cdef uintptr_t Tx_ptr = Tx.ctypes.data
        batched_jones_transform(
            handle_[0], arima_mem_ptr[0], order, <int> self.batch_size,
            <bool> isInv, <double*>x_ptr, <double*>Tx_ptr)

        del arima_mem_ptr

        return (Tx)
