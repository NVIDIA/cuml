# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import numpy as np

from cuml.internals.base import Base, get_handle
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import check_array
from cuml.tsa._deprecation import warn_deprecated_tsa_api

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/tsa/holtwinters_params.h" namespace "ML" nogil:
    enum SeasonalType:
        ADDITIVE
        MULTIPLICATIVE

cdef extern from "cuml/tsa/holtwinters.h" namespace "ML::HoltWinters" nogil:
    cdef void buffer_size(
        int n, int batch_size, int frequency,
        int *start_leveltrend_len, int *start_season_len,
        int *components_len, int *error_len,
        int *leveltrend_coef_shift, int *season_coef_shift) except +

    cdef void fit(
        handle_t &handle, int n, int batch_size,
        int frequency, int start_periods, SeasonalType seasonal,
        float epsilon,
        float *data, float *level_ptr, float *trend_ptr,
        float *season_ptr, float *SSE_error_ptr) except +
    cdef void fit(
        handle_t &handle, int n, int batch_size,
        int frequency, int start_periods, SeasonalType seasonal,
        double epsilon,
        double *data, double *level_ptr, double *trend_ptr,
        double *season_ptr, double *SSE_error_ptr) except +

    cdef void forecast(
        handle_t &handle, int n, int batch_size, int frequency,
        int h, SeasonalType seasonal, float *level_ptr,
        float *trend_ptr, float *season_ptr, float *forecast_ptr) except +
    cdef void forecast(
        handle_t &handle, int n, int batch_size, int frequency,
        int h, SeasonalType seasonal, double *level_ptr,
        double *trend_ptr, double *season_ptr, double *forecast_ptr) except +


class ExponentialSmoothing(Base):
    """
    Implements a HoltWinters time series analysis model which is used in
    both forecasting future entries in a time series as well as in providing
    exponential smoothing, where weights are assigned against historical
    data with exponentially decreasing impact. This is done by analyzing
    three components of the data: level, trend, and seasonality.

    .. deprecated:: 26.08
        ``cuml.tsa.ExponentialSmoothing`` and ``cuml.ExponentialSmoothing``
        are deprecated and will be removed in the cuML 26.12 release.

    Notes
    -----
    *Known Limitations:* This version of ExponentialSmoothing currently
    provides only a limited number of features when compared to the
    `statsmodels.holtwinters.ExponentialSmoothing` model. Noticeably, it lacks:

    * predict : no support for in-sample prediction.
        * https://github.com/NVIDIA/cuml/issues/875

    * hessian : no support for returning Hessian matrix.
        * https://github.com/NVIDIA/cuml/issues/880

    * information : no support for returning Fisher matrix.
        * https://github.com/NVIDIA/cuml/issues/880

    * loglike : no support for returning Log-likelihood.
        * https://github.com/NVIDIA/cuml/issues/880

    Additionally, be warned that there may exist floating point instability
    issues in this model. Small values in endog may lead to faulty results.
    See https://github.com/NVIDIA/cuml/issues/888 for more information.

    *Known Differences:* This version of ExponentialSmoothing differs from
    statsmodels in some other minor ways:

    * Cannot pass trend component or damped trend component
    * this version can take additional parameters `eps`,
      `start_periods`, and `ts_num`
    * Score returns SSE rather than gradient logL
      https://github.com/NVIDIA/cuml/issues/876
    * This version provides get_level(), get_trend(), get_season()

    Examples
    --------

    .. code-block:: python

        >>> from cuml import ExponentialSmoothing
        >>> import cudf
        >>> import cupy as cp
        >>> data = cudf.Series([1, 2, 3, 4, 5, 6,
        ...                     7, 8, 9, 10, 11, 12,
        ...                     2, 3, 4, 5, 6, 7,
        ...                     8, 9, 10, 11, 12, 13,
        ...                     3, 4, 5, 6, 7, 8, 9,
        ...                     10, 11, 12, 13, 14],
        ...                     dtype=cp.float64)
        >>> cu_hw = ExponentialSmoothing(data, seasonal_periods=12).fit()
        >>> cu_pred = cu_hw.forecast(4)
        >>> print('Forecasted points:', cu_pred) # doctest: +SKIP
        Forecasted points :
        0    4.000143766093652
        1    5.000000163513641
        2    6.000000000174092
        3    7.000000000000178

    Parameters
    ----------
    endog : array-like (device or host)
        Acceptable formats: cuDF DataFrame, cuDF Series,
        NumPy ndarray, Numba device ndarray, cuda array interface
        compliant array like CuPy.
        Note: cuDF.DataFrame types assumes data is in columns,
        while all other datatypes assume data is in rows.
        The endogenous dataset to be operated on.
    seasonal : 'additive', 'add', 'multiplicative', 'mul' \
        (default = 'additive')
        Whether the seasonal trend should be calculated
        additively or multiplicatively.
    seasonal_periods : int (default=2)
        The seasonality of the data (how often it
        repeats). For monthly data this should be 12,
        for weekly data, this should be 7.
    start_periods : int (default=2)
        Number of seasons to be used for seasonal seed values
    ts_num : int (default=1)
        The number of different time series that were passed
        in the endog param.
    eps : np.number > 0 (default=2.24e-3)
        The accuracy to which gradient descent should achieve.
        Note that changing this value may affect the forecasted results.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    """

    forecasted_points = ReflectedAttr()
    level = ReflectedAttr()
    trend = ReflectedAttr()
    season = ReflectedAttr()
    SSE = ReflectedAttr()

    def __init__(self, endog, *, seasonal="additive",
                 seasonal_periods=2, start_periods=2,
                 ts_num=1, eps=2.24e-3,
                 verbose=False, output_type=None):

        warn_deprecated_tsa_api("cuml.tsa.ExponentialSmoothing")

        super().__init__(verbose=verbose, output_type=output_type)

        # Total number of Time Series for forecasting
        if not isinstance(ts_num, int):
            raise TypeError("Type of ts_num must be int. Given: " +
                            type(ts_num))
        if ts_num <= 0:
            raise ValueError("Must state at least 1 series. Given: " +
                             str(ts_num))
        self.ts_num = ts_num

        # Season length in the time series
        if not isinstance(seasonal_periods, int):
            raise TypeError("Type of seasonal_periods must be int."
                            " Given: " + type(seasonal_periods))
        if seasonal_periods < 2:
            raise ValueError("Frequency must be >= 2. Given: " +
                             str(seasonal_periods))
        self.seasonal_periods = seasonal_periods

        # whether to perform additive or multiplicative STL decomposition
        if seasonal in ["additive", "add"]:
            self.seasonal = "add"
            self._cpp_stype = ADDITIVE
        elif seasonal in ["multiplicative", "mul"]:
            self.seasonal = "mul"
            self._cpp_stype = MULTIPLICATIVE
        else:
            raise ValueError("Seasonal must be either "
                             "\"additive\" or \"multiplicative\".")

        # number of seasons to be used for seasonal seed values
        if not isinstance(start_periods, int):
            raise TypeError("Type of start_periods must be int. Given: " +
                            type(start_periods))
        if start_periods < 2:
            raise ValueError("Start Periods must be >= 2. Given: " +
                             str(start_periods))
        if seasonal_periods < start_periods:
            raise ValueError("Seasonal_Periods (" + str(seasonal_periods) +
                             ") cannot be less than start_periods (" +
                             str(start_periods) + ").")
        self.start_periods = start_periods

        if not np.issubdtype(type(eps), np.number):
            raise TypeError("Epsilon provided is of type " + type(eps) +
                            " and thus cannot be cast to float() or double()")
        if eps <= 0:
            raise ValueError("Epsilon must be positive. Given: " + eps)

        # Set up attributes:
        self.eps = eps
        self.endog = endog
        self.forecasted_points = []  # list for final forecast output
        self.level = []  # list for level values for each time series in batch
        self.trend = []  # list for trend values for each time series in batch
        self.season = []  # list for season values for each series in batch
        self.SSE = []     # SSE for all time series in batch
        self.fit_executed_flag = False
        self.h = 0

    def _check_dims(self, ts_input, is_cudf=False):
        err_mess = ("ExponentialSmoothing initialized with "
                    + str(self.ts_num) +
                    " time series, but data has dimension ")

        is_cudf = isinstance(ts_input, cudf.DataFrame)

        mod_ts_input = check_array(
            ts_input,
            order="C",
            ensure_2d=False,
            ensure_all_finite=False,
        )

        if len(mod_ts_input.shape) == 1:
            self.n = mod_ts_input.shape[0]
            if self.ts_num != 1:
                raise ValueError(err_mess + "1.")
        elif len(ts_input.shape) == 2:
            if(is_cudf):
                d1 = mod_ts_input.shape[0]
                d2 = mod_ts_input.shape[1]
                mod_ts_input = mod_ts_input.reshape((d1*d2,))
            else:
                d1 = mod_ts_input.shape[1]
                d2 = mod_ts_input.shape[0]
                mod_ts_input = mod_ts_input.ravel()
            self.n = d1
            if self.ts_num != d2:
                raise ValueError(err_mess + str(d2))
        else:
            raise ValueError("Data input must have 1 or 2 dimensions.")
        return mod_ts_input

    @mlfunc(convert_output=False)
    def fit(self) -> "ExponentialSmoothing":
        """
        Perform fitting on the given `endog` dataset.
        Calculates the level, trend, season, and SSE components.
        """

        X = self._check_dims(self.endog)

        self.dtype = X.dtype

        if self.n < self.start_periods*self.seasonal_periods:
            raise ValueError("Length of time series (" + str(self.n) +
                             ") must be at least freq*start_periods (" +
                             str(self.start_periods*self.seasonal_periods) +
                             ").")
        if self.n <= 0:
            raise ValueError("Time series must contain at least 1 value."
                             " Given: " + str(self.n))

        cdef uintptr_t input_ptr
        cdef int leveltrend_seed_len, season_seed_len, components_len
        cdef int leveltrend_coef_offset, season_coef_offset
        cdef int error_len

        input_ptr = X.data.ptr

        buffer_size(<int> self.n, <int> self.ts_num,
                    <int> self.seasonal_periods,
                    <int*> &leveltrend_seed_len,
                    <int*> &season_seed_len,
                    <int*> &components_len,
                    <int*> &leveltrend_coef_offset,
                    <int*> &season_coef_offset,
                    <int*> &error_len)

        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef uintptr_t level_ptr, trend_ptr, season_ptr, SSE_ptr

        level = cp.zeros(components_len, dtype=self.dtype)
        trend = cp.zeros(components_len, dtype=self.dtype)
        season = cp.zeros(components_len, dtype=self.dtype)
        SSE = cp.zeros(self.ts_num, dtype=self.dtype)

        level_ptr = level.data.ptr
        trend_ptr = trend.data.ptr
        season_ptr = season.data.ptr
        SSE_ptr = SSE.data.ptr

        cdef float eps_f = np.float32(self.eps)
        cdef double eps_d = np.float64(self.eps)

        if self.dtype == np.float32:
            fit(handle_[0], <int> self.n, <int> self.ts_num,
                <int> self.seasonal_periods, <int> self.start_periods,
                <SeasonalType> self._cpp_stype,
                <float> eps_f,
                <float*> input_ptr, <float*> level_ptr,
                <float*> trend_ptr, <float*> season_ptr,
                <float*> SSE_ptr)

        elif self.dtype == np.float64:
            fit(handle_[0], <int> self.n, <int> self.ts_num,
                <int> self.seasonal_periods, <int> self.start_periods,
                <SeasonalType> self._cpp_stype,
                <double> eps_d,
                <double*> input_ptr, <double*> level_ptr,
                <double*> trend_ptr, <double*> season_ptr,
                <double*> SSE_ptr)

        else:
            raise TypeError("ExponentialSmoothing supports only float32"
                            " and float64 input, but input type "
                            + str(self.dtype) + " passed.")
        handle.sync()

        num_rows = int(components_len/self.ts_num)
        self.level = level.reshape((self.ts_num, num_rows), order='F')
        self.trend = trend.reshape((self.ts_num, num_rows), order='F')
        self.season = season.reshape((self.ts_num, num_rows), order='F')
        self.SSE = SSE
        self.fit_executed_flag = True

        return self

    @mlfunc(convert_output=False)
    def forecast(self, h=1, index=None):
        """
        Forecasts future points based on the fitted model.

        Parameters
        ----------
        h : int (default=1)
            The number of points for each series to be forecasted.
        index : int (default=None)
            The index of the time series from which you want
            forecasted points. if None, then a cudf.DataFrame of
            the forecasted points from all time series is returned.

        Returns
        -------
        preds : cudf.DataFrame or cudf.Series
            Series of forecasted points if index is provided.
            DataFrame of all forecasted points if index=None.

        """
        cdef uintptr_t forecast_ptr, level_ptr, trend_ptr, season_ptr
        handle = get_handle()
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        if not isinstance(h, int) or (not isinstance(index, int) and index is not None):
            raise TypeError("Input arguments must be of type int."
                            "Index has type: " + str(type(index))
                            + "\nh has type: " + str(type(h)))

        if self.fit_executed_flag:
            if h <= 0:
                raise ValueError("h must be > 0. Currently: " + str(h))

            if h > self.h:
                self.h = h
                self.forecasted_points = cp.zeros(
                    (self.ts_num, h), dtype=self.dtype, order="F"
                )
                forecast_ptr = self.forecasted_points.data.ptr
                level_ptr = self.level.data.ptr
                trend_ptr = self.trend.data.ptr
                season_ptr = self.season.data.ptr

                if self.dtype == np.float32:
                    forecast(handle_[0], <int> self.n,
                             <int> self.ts_num,
                             <int> self.seasonal_periods,
                             <int> h,
                             <SeasonalType> self._cpp_stype,
                             <float*> level_ptr,
                             <float*> trend_ptr,
                             <float*> season_ptr,
                             <float*> forecast_ptr)
                elif self.dtype == np.float64:
                    forecast(handle_[0], <int> self.n,
                             <int> self.ts_num,
                             <int> self.seasonal_periods, <int> h,
                             <SeasonalType> self._cpp_stype,
                             <double*> level_ptr,
                             <double*> trend_ptr,
                             <double*> season_ptr,
                             <double*> forecast_ptr)

                handle.sync()

            if index is None:
                if self.ts_num == 1:
                    return cudf.Series(self.forecasted_points.ravel(order='F')[:h])
                else:
                    return cudf.DataFrame(self.forecasted_points[:, :h].T)
            else:
                if index < 0 or index >= self.ts_num:
                    raise IndexError("Index input: " + str(index) +
                                     " outside of range [0, " +
                                     str(self.ts_num) + "]")
                return cudf.Series(self.forecasted_points[index, :h])
        else:
            raise ValueError("Fit() the model before forecast()")

    @mlfunc(convert_output=False)
    def score(self, index=None):
        """
        Returns the score of the model.

        .. note:: Currently returns the SSE, rather than the gradient of the
            LogLikelihood. https://github.com/NVIDIA/cuml/issues/876

        Parameters
        ----------
        index : int (default=None)
            The index of the time series from which the SSE will be
            returned. if None, then all SSEs are returned in a cudf
            Series.

        Returns
        -------
        score : np.float32, np.float64, or cudf.Series
            The SSE of the fitted model.

        """
        if self.fit_executed_flag:
            if index is None:
                return cudf.Series(self.SSE)
            elif index < 0 or index >= self.ts_num:
                raise IndexError("Index input: " + str(index) + " outside of "
                                 "range [0, " + str(self.ts_num) + "]")
            else:
                return self.SSE[index]
        else:
            raise ValueError("Fit() the model before score()")

    @mlfunc(convert_output=False)
    def get_level(self, index=None):
        """
        Returns the level component of the model.

        Parameters
        ----------
        index : int (default=None)
            The index of the time series from which the level will be
            returned. if None, then all level components are returned
            in a cudf.Series.

        Returns
        -------
        level : cudf.Series or cudf.DataFrame
            The level component of the fitted model
        """
        if self.fit_executed_flag:
            if index is None:
                if self.ts_num == 1:
                    return cudf.Series(self.level.ravel(order='F'))
                else:
                    return cudf.DataFrame(self.level.T)
            else:
                if index < 0 or index >= self.ts_num:
                    raise IndexError("Index input: " + str(index) + " outside "
                                     "of range [0, " + str(self.ts_num) + "]")
                else:
                    return cudf.Series(cp.asarray(self.level[index]))
        else:
            raise ValueError("Fit() the model to get level values")

    @mlfunc(convert_output=False)
    def get_trend(self, index=None):
        """
        Returns the trend component of the model.

        Parameters
        ----------
        index : int (default=None)
            The index of the time series from which the trend will be
            returned. if None, then all trend components are returned
            in a cudf.Series.

        Returns
        -------
        trend : cudf.Series or cudf.DataFrame
            The trend component of the fitted model.
        """
        if self.fit_executed_flag:
            if index is None:
                if self.ts_num == 1:
                    return cudf.Series(self.trend.ravel(order='F'))
                else:
                    return cudf.DataFrame(self.trend.T)
            else:
                if index < 0 or index >= self.ts_num:
                    raise IndexError("Index input: " + str(index) + " outside "
                                     "of range [0, " + str(self.ts_num) + "]")
                else:
                    return cudf.Series(cp.asarray(self.trend[index]))
        else:
            raise ValueError("Fit() the model to get trend values")

    @mlfunc(convert_output=False)
    def get_season(self, index=None):
        """
        Returns the season component of the model.

        Parameters
        ----------
        index : int (default=None)
            The index of the time series from which the season will be
            returned. if None, then all season components are returned
            in a cudf.Series.

        Returns
        -------
        season: cudf.Series or cudf.DataFrame
            The season component of the fitted model
        """
        if self.fit_executed_flag:
            if index is None:
                if self.ts_num == 1:
                    return cudf.Series(self.season.ravel(order='F'))
                else:
                    return cudf.DataFrame(self.season.T)
            else:
                if index < 0 or index >= self.ts_num:
                    raise IndexError("Index input: " + str(index) + " outside "
                                     "of range [0, " + str(self.ts_num) + "]")
                else:
                    return cudf.Series(cp.asarray(self.season[index]))
        else:
            raise ValueError("Fit() the model to get season values")

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "endog",
            "seasonal",
            "seasonal_periods",
            "start_periods",
            "ts_num",
            "eps",
        ]
