#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import numbers
import warnings

import cudf
import cudf.pandas
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from packaging.version import Version
from pandas.api.types import is_string_dtype
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_is_fitted

__all__ = (
    "check_is_fitted",
    "check_random_seed",
    "check_features",
    "check_input_features",
    "check_consistent_length",
    "check_all_finite",
    "check_non_negative",
    "check_array",
    "check_cudf",
    "check_y",
    "check_sample_weight",
    "check_inputs",
    "check_classification_targets",
)

_CUPY_SUPPORTS_LARGE_SPARSE = Version(cp.__version__) >= Version("14.1.0")


def check_random_seed(random_state) -> int:
    """Turn a `random_state` argument into a seed.

    Parameters
    ----------
    random_state : None | int | instance of RandomState
        If random_state is None, return a random int as seed.
        If random_state is an int, return it.
        If random_state is a RandomState instance, derive a seed from it.

    Returns
    -------
    seed : int
        A seed in the range [0, 2**32 - 1].
    """
    if isinstance(random_state, numbers.Integral):
        if random_state < 0 or random_state >= 2**32:
            raise ValueError(
                f"Expected `0 <= random_state <= 2**32 - 1`, got {random_state}"
            )
        return int(random_state)

    if random_state is None:
        randint = np.random.randint
    elif isinstance(
        random_state, (np.random.RandomState, cp.random.RandomState)
    ):
        randint = random_state.randint
    else:
        raise TypeError(
            f"`random_state` must be an `int`, an instance of `RandomState`, or `None`. "
            f"Got {random_state!r} instead."
        )

    # randint returns in [low, high), so high=2**32 to sample all uint32s
    return int(randint(low=0, high=2**32, dtype=np.uint32))


def _check_shape(
    shape,
    *,
    ensure_2d=True,
    ensure_min_samples=0,
    ensure_min_features=0,
    array_type=None,
) -> None:
    """Check that an input shape is as expected.

    Parameters
    ----------
    shape : tuple
        The array shape.
    ensure_2d : bool, default=True
        If True, only 2D arrays are accepted. Otherwise accepts 1D or 2D
        arrays.
    ensure_min_samples : int, default=0
        A minimum number of samples to require. Defaults to 0 for no minimum.
    ensure_min_features : int, default=0
        A minimum number of features to require. Defaults to 0 for no minimum.
    array_type : type or None, default=None
        The type of the array-like object. Used in error messages.
    """
    ndim = len(shape)

    if ndim == 0 or ndim == 1 and ensure_2d:
        if issubclass(array_type, (cudf.Series, pd.Series)):
            msg = (
                f"Expected a 2-dimensional container but got {array_type} "
                "instead. Pass a DataFrame containing a single row (i.e. "
                "single sample) or a single column (i.e. single feature) "
                "instead."
            )
        else:
            kind = "scalar" if ndim == 0 else "1D"
            msg = (
                f"Expected 2D array, got {kind} array instead. Reshape your data "
                "using array.reshape(-1, 1) if your data has a single feature, "
                "or array.reshape(1, -1) if it contains a single sample."
            )
        raise ValueError(msg)
    elif ndim > 2:
        raise ValueError(f"Expected 2D array, got {ndim}D array instead.")

    if ensure_min_samples > 0:
        n_samples = shape[0]
        if n_samples < ensure_min_samples:
            raise ValueError(
                f"Found array with {n_samples} sample(s) (shape={shape}) "
                f"while a minimum of {ensure_min_samples} is required."
            )

    if ensure_min_features > 0 and ndim == 2:
        n_features = shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                f"Found array with {n_features} feature(s) (shape={shape}) "
                f"while a minimum of {ensure_min_features} is required."
            )


def _get_n_features(X):
    """Get the number of features in X."""
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            return 0
        row = X[0]
        # For non-array inputs, we assume that all nested lists have the same
        # length. This matches sklearn's implementation as well. If this
        # assumption isn't true, then later validation code will error anyway.
        # We only take the length of sub-sequences that numpy wouldn't treat as
        # single elements.
        if not isinstance(row, (str, bytes, dict)):
            try:
                return len(row)
            except Exception:
                pass

    if hasattr(X, "shape"):
        shape = X.shape
    elif hasattr(X, "__cuda_array_interface__"):
        shape = X.__cuda_array_interface__["shape"]
    elif hasattr(X, "__array_interface__"):
        shape = X.__array_interface__["shape"]
    else:
        shape = np.asarray(X).shape

    _check_shape(shape, ensure_2d=True, array_type=type(X))

    return shape[1]


def _get_n_samples(X):
    """Get the number of samples in X."""

    if (shape := getattr(X, "shape", None)) is not None:
        if len(shape) == 0:
            raise TypeError("Expected array-like, got scalar instead.")
        return shape[0]

    try:
        return len(X)
    except TypeError as exc:
        raise TypeError(
            f"Expected array-like, got {type(X)} instead."
        ) from exc


def _get_feature_names(X):
    """Get feature names from X.

    Returns
    -------
    names: ndarray or None
        Feature names of `X`. Unrecognized array containers will return `None`.
    """
    if isinstance(X, (pd.DataFrame, cudf.DataFrame)):
        feature_names = np.asarray(X.columns, dtype=object)
    elif hasattr(X, "__dataframe__"):
        pandas4_warning = getattr(pd.errors, "Pandas4Warning", FutureWarning)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The Dataframe Interchange Protocol is deprecated.*",
                category=pandas4_warning,
            )
            feature_names = np.asarray(
                list(X.__dataframe__().column_names()), dtype=object
            )
    else:
        return None

    if len(feature_names) == 0:
        # No features, just return None
        return None

    # Check the types of the column names.
    types = sorted(t.__qualname__ for t in set(type(v) for v in feature_names))
    if len(types) == 1 and types[0] == "str":
        return feature_names
    elif len(types) > 1 and "str" in types:
        raise TypeError(
            "Feature names are only supported if all input features have string names, "
            f"but your input has {types} as feature name / column name types. "
            "If you want feature names to be stored and validated, you must convert "
            "them all to strings, by using X.columns = X.columns.astype(str) for "
            "example. Otherwise you can remove feature / column names from your input "
            "data, or convert them all to a non-string data type."
        )

    return None


def check_features(estimator, X, reset=False) -> None:
    """Check or set ``n_features_in_`` and ``feature_names_in_``.

    Parameters
    ----------
    estimator : Base
        The estimator to check.
    X : array-like
        The original user-provided `X` input. No conversion or processing steps
        should have occurred to this array yet.
    reset : bool, default=False
        If True, ``n_features_in_`` and ``feature_names_in_`` are set on
        ``estimator`` to match ``X``. Otherwise ``X`` is checked to match the
        existing ``n_features_in_`` and ``feature_names_in_``. ``reset=True``
        should be used for fit-like methods, and False otherwise.
    """
    n_features = _get_n_features(X)
    feature_names = _get_feature_names(X)

    if reset:
        estimator.n_features_in_ = n_features
        if feature_names is not None:
            estimator.feature_names_in_ = feature_names
        elif hasattr(estimator, "feature_names_in_"):
            # Clear old feature names if present
            delattr(estimator, "feature_names_in_")
        return

    est_feature_names = getattr(estimator, "feature_names_in_", None)

    # Check feature_names_in_ first
    if est_feature_names is not None or feature_names is not None:
        if est_feature_names is None:
            warnings.warn(
                f"X has feature names, but {estimator.__class__.__name__} was fitted "
                "without feature names"
            )

        elif feature_names is None:
            warnings.warn(
                "X does not have valid feature names, but"
                f" {estimator.__class__.__name__} was fitted with feature names"
            )

        elif len(est_feature_names) != len(feature_names) or np.any(
            est_feature_names != feature_names
        ):
            unexpected = sorted(
                set(feature_names).difference(est_feature_names)
            )
            missing = sorted(set(est_feature_names).difference(feature_names))

            parts = [
                "The feature names should match those that were passed during fit."
            ]
            for heading, names in [
                ("Feature names unseen at fit time:", unexpected),
                ("Feature names seen at fit time, yet now missing:", missing),
            ]:
                if names:
                    parts.append(heading)
                    parts.extend([f"- {name}" for name in names[:5]])
                    if len(names) > 5:
                        parts.append("- ...")

            if not missing and not unexpected:
                parts.append(
                    "Feature names must be in the same order as they were in fit."
                )

            msg = "\n".join(parts)
            raise ValueError(msg)

    # Then check n_features_in_. We skip the check if `n_features_in_` wasn't
    # stored (through calling `fit` first). This is allowed for some stateless
    # transformers.
    if (
        hasattr(estimator, "n_features_in_")
        and n_features != estimator.n_features_in_
    ):
        raise ValueError(
            f"X has {n_features} features, but {estimator.__class__.__name__} "
            f"is expecting {estimator.n_features_in_} features as input."
        )


def check_input_features(estimator, input_features=None):
    """Check `input_features` and generate names if needed.

    Mainly useful when implementing `get_feature_names_out`.

    Parameters
    ----------
    input_features : array-like of str or None, default=None
        Input features.

        - If `input_features` is `None`, then `feature_names_in_` is used as
          input feature names. If `feature_names_in_` is not defined, then the
          following input feature names are generated: `["x0", "x1", ...,
          "x(n_features_in_ - 1)"]`.
        - If `input_features` is an array-like, then `input_features` must
          match `feature_names_in_` if `feature_names_in_` is defined.

    Returns
    -------
    feature_names_in : numpy.ndarray[str] or None
        Validated input feature names.
    """
    feature_names_in = getattr(estimator, "feature_names_in_", None)

    if input_features is not None:
        input_features = np.asarray(input_features, dtype=object)
        if feature_names_in is not None and not np.array_equal(
            feature_names_in, input_features
        ):
            raise ValueError(
                "input_features is not equal to feature_names_in_"
            )

        elif len(input_features) != estimator.n_features_in_:
            raise ValueError(
                "input_features should have length equal to number of "
                f"features ({estimator.n_features_in_}), got {len(input_features)}"
            )
        return input_features

    if feature_names_in is not None:
        return feature_names_in

    return np.asarray(
        [f"x{i}" for i in range(estimator.n_features_in_)], dtype=object
    )


def check_consistent_length(*arrays) -> None:
    """Check whether all inputs have the same number of samples.

    Typically should be called after arrays have been validated and normalized
    by other checks, but also works on typical unvetted user inputs.

    Parameters
    ----------
    *arrays : array or None
        The input variables to validate. None-values are ignored.
    """
    lengths = [_get_n_samples(X) for X in arrays if X is not None]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"Found input variables with inconsistent number of samples: "
            f"{sorted(n for n in lengths)}"
        )


# Returns status in a bitfield:
# 0b01: contains NaN
# 0b10: contains +/-inf
_cupy_any_inf_or_nan = cp.ReductionKernel(
    "T x",
    "uint8 out",
    "((unsigned char)isinf(x)) << 1 | ((unsigned char)isnan(x))",
    "a | b",
    "out = a",
    "0",
    "any_inf_or_nan",
)


def check_all_finite(array, *, allow_nan=False, input_name=None) -> None:
    """Check if all input values are finite.

    This check is skipped if scikit-learn's ``assume_finite`` option is
    configured via ``sklearn.set_config(assume_finite=True)``.

    Parameters
    ----------
    array : dense or sparse array
        The array to check.
    allow_nan : bool, default=False
        Whether to allow NaN values.
    input_name : str or None, default=None
        The input parameter name to use in error messages.
    """
    if not array.dtype.kind == "f":
        # No-op for non floating inputs
        return

    if sklearn.get_config()["assume_finite"]:
        # no-op if assume_finite configured
        return

    if cp_sp.issparse(array) or sp.issparse(array):
        array = array.data

    if not array.size:
        # No-op for empty inputs
        return

    has_nan = has_inf = False
    if isinstance(array, cp.ndarray):
        status = _cupy_any_inf_or_nan(array).item()
        has_nan = status & 0b01
        has_inf = status & 0b10
    else:
        # First try an O(1) space solution for the common case
        with np.errstate(over="ignore", invalid="ignore"):
            x_sum = array.sum()
        if not np.isfinite(x_sum):
            # We can't infer anything from the value of x_sum being non-finite
            # - NaN could mean NaN present, or both -inf and inf
            # - inf could mean inf present, or just overflow
            # Here we selectively apply O(n) space fallbacks as needed.
            if allow_nan:
                has_inf = np.isinf(array).any()
            elif not (has_nan := np.isnan(array).any()):
                has_inf = np.isinf(array).any()

    if has_nan and not allow_nan:
        msg = "NaN"
    elif has_inf:
        msg = f"infinity or a value too large for {array.dtype!r}"
    else:
        msg = None

    if msg is not None:
        raise ValueError(f"Input {input_name or 'array'} contains {msg}.")


def check_non_negative(array, *, input_name=None) -> None:
    """Check if all input values are non-negative.

    Parameters
    ----------
    array : dense or sparse array
        The array to check.
    input_name : str or None, default=None
        The input parameter name to use in error messages.
    """
    if cp_sp.issparse(array) or sp.issparse(array):
        array = array.data
    xp = cp if isinstance(array, cp.ndarray) else np
    if array.size != 0 and xp.nanmin(array) < 0:
        suffix = f" passed to {input_name}" if input_name is not None else ""
        raise ValueError(f"Negative values in data{suffix}")


def _requires_int64_sparse(array):
    """Check if a sparse array requires int64 indices"""
    INT32_MAX = (1 << 31) - 1
    # A sparse array requires int64 indices if:
    # - It has shape or nnz that doesn't fit in an int32
    # - CSR/CSC/BSR have indices/indptr that don't fit in an int32
    return (
        any(s > INT32_MAX for s in array.shape)
        or array.nnz > INT32_MAX
        or (
            array.format in ["csr", "csc", "bsr"]
            and (
                len(array.indices) > INT32_MAX or len(array.indptr) > INT32_MAX
            )
        )
    )


def _ensure_int32_sparse(array):
    """Convert sparse array to int32 indices if possible, and error otherwise"""
    if _requires_int64_sparse(array):
        raise ValueError(
            "Only sparse matrices with int32 indices are currently supported."
        )

    # Definitely safe to downscast to int32, but only cast if needed since
    # the sparse constructors do a little bit of work.
    if array.format == "coo":
        if array.row.dtype == "int32" and array.col.dtype == "int32":
            return array
        return type(array)(
            (
                array.data,
                (
                    array.row.astype("int32", copy=False),
                    array.col.astype("int32", copy=False),
                ),
            ),
            shape=array.shape,
        )
    elif array.format == "dia":
        if array.offsets.dtype == "int32":
            return array
        return type(array)(
            (array.data, array.offsets.astype("int32")), shape=array.shape
        )
    elif array.format in ["csr", "csc", "bsr"]:
        if array.indices.dtype == "int32" and array.indptr.dtype == "int32":
            return array
        return type(array)(
            (
                array.data,
                array.indices.astype("int32", copy=False),
                array.indptr.astype("int32", copy=False),
            ),
            shape=array.shape,
        )
    else:
        # Other type without numeric indices, can just return
        return array


def _index_as_mem_type(index, mem_type=None):
    """Coerce `index` to a specified `mem_type` (if needed)."""
    if isinstance(index, cudf.Index) and mem_type == "host":
        index = (
            cudf.pandas.as_proxy_object(index)
            if cudf.pandas.LOADED
            else index.to_pandas()
        )
    elif isinstance(index, pd.Index) and mem_type == "device":
        index = cudf.Index(index)
    return index


if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
    np_asarray = np.asarray
else:

    def np_asarray(x, dtype=None, order=None, copy=None):
        """A compatibility shim for `np.asarray`.

        numpy 2.0 added the `copy` arg to `np.asarray`, as well as changed the
        meaning of copy=False to "error if a copy required" rather than "only
        copy if needed" (which is now `copy=None`)."""
        return np.array(x, dtype=dtype, order=order or "K", copy=bool(copy))


def check_array(
    array,
    *,
    accept_sparse=False,
    accept_large_sparse=False,
    dtype=None,
    mem_type="device",
    order="A",
    copy=False,
    ensure_all_finite=True,
    ensure_non_negative=False,
    ensure_2d=True,
    ensure_min_samples=1,
    ensure_min_features=1,
    input_name=None,
    return_index=False,
):
    """Validate and coerce an array-like to a supported type.

    Parameters
    ----------
    array : array-like
        The array-like input to validate.
    accept_sparse : bool, str, list[str], default=False
        The sparse matrix format(s) to support. If the input is sparse
        but not in a supported format, it will be converted to the first
        listed format. Pass True to support any input format. The default
        of False will raise an error on sparse inputs.
    accept_large_sparse : bool, default=False
        Whether large (int64) indices are supported for sparse containers with
        CSR/CSC/COO/BSR formats. If not supported, an appropriate error will be
        raised if the sparse indices aren't int32.
    dtype : None, dtype, list[dtype], default=None
        The dtype(s) to support. By default no dtype validation is performed.
        Pass a dtype or a list of supported dtypes to enforce a dtype for the
        output. If the input doesn't have a supported dtype, it will be
        converted to the first listed dtype.
    mem_type : {'device', 'host'} or None, default='device'
        The memory type use for the output. If 'device', the output will be a
        ``cupy.ndarray`` if dense, or a ``cupyx.scipy.sparse.spmatrix`` if
        sparse. If 'host', the output will be a ``numpy.ndarray`` if dense, or
        a ``scipy.sparse.spmatrix`` if sparse. If ``None``, the output will
        have the same memory type as the input (i.e. device if already on
        device, host otherwise).
    order : {'F', 'C', 'A', None}, default='A'
        The order and contiguity to enforce for dense outputs. Use 'F' for
        F-contiguous outputs, 'C' for C-contiguous outputs, 'A' for either F or
        C contiguous, or `None` for no contiguity requirements (may be
        non-contiguous!).
    copy : bool, default=False
        Set to ``True`` to ensure the input is copied (allowing the output to
        be mutated without worry). Note that the default of ``False`` doesn't
        guarantee no copy is made, it only allows for zero-copy when possible.
    ensure_all_finite : bool or 'allow-nan', default=True
        If True, an error will be raised if non-finite values are found in the
        input. If 'allow-nan', an error will be raised if infinite values are
        found (but not for NaN). If False then ``check_all_finite`` is skipped.
    ensure_non_negative : bool, default=False
        If True, an error will be raised if negative values are found in the
        input. By default ``check_non_negative`` is skipped.
    ensure_2d : bool, default=True
        If True, the input must be 2D. If False, 1D or 2D inputs are accepted.
    ensure_min_samples : int, default=1
        A minimum number of samples to require. Set to 0 for no minimum.
    ensure_min_features : int, default=1
        A minimum number of features to require for 2D inputs. Set to 0 for no
        minimum.
    input_name : str or None, default=None
        The input parameter name to use in error messages.
    return_index : bool, default=False
        Whether to return the index of ``array`` (if a dataframe-like value).
        This is useful for functions that need to return an output with a
        dataframe index aligned with the input.

    Returns
    -------
    array : dense or sparse array
        The converted and validated array. Depending on input and parameters,
        will be one of ``cupy.ndarray``, ``numpy.ndarray``,
        ``cupyx.scipy.sparse.spmatrix``, or ``scipy.sparse.spmatrix``.
    index : pandas.Index, cudf.Index, or None
        The index of the input if a dataframe-like, or None if no index. The
        index will be converted to match ``mem_type``. Only returned if
        ``return_index=True``.
    """
    # Normalize and validate arguments
    if mem_type not in ("device", "host", None):
        raise ValueError(f"Unsupported {mem_type=!r}")
    if order not in ("F", "C", "A", None):
        raise ValueError(f"Unsupported {order=!r}")

    if dtype is not None:
        if not isinstance(dtype, (list, tuple)):
            dtype = [dtype]
        dtype = [np.dtype(i) for i in dtype]

    is_sparse = cp_sp.issparse(array) or sp.issparse(array)
    if is_sparse and (
        mem_type == "device" or (mem_type is None and cp_sp.issparse(array))
    ):
        # XXX: cupyx.scipy.sparse doesn't support integral dtypes. If a dtype
        # is specified, we filter to only supported types (erroring if no
        # supported types specified). If dtype=None, we use the input dtype if
        # supported, and the closest floating type otherwise.
        if dtype is not None:
            if not any(d.kind in "fb" for d in dtype):
                raise ValueError(
                    f"No dtype in {dtype} is supported by cupyx.scipy.sparse"
                )
            dtype = [d for d in dtype if d.kind in "fb"]
        elif array.dtype.kind not in "fb":
            dtype = [
                np.dtype("f4") if array.dtype.itemsize <= 4 else np.dtype("f8")
            ]

    # Extract original array type and dtype (when possible)
    array_type = type(array)
    if isinstance(array, (cudf.DataFrame, pd.DataFrame)):
        # Unify dataframe dtypes when possible, None otherwise
        if all(isinstance(dt, np.dtype) for dt in array.dtypes):
            array_dtype = np.result_type(*array.dtypes)
        elif any(dt == "object" or is_string_dtype(dt) for dt in array.dtypes):
            array_dtype = np.dtype("object")
        else:
            array_dtype = None
    else:
        # Extract and normalize the `dtype` attribute if present. If `dtype` is
        # present but doesn't coerce to a numpy dtype, fall back to None. For
        # example, pytorch arrays have a non-numpy-compatible `dtype` attr.
        array_dtype = getattr(array, "dtype", None)
        if array_dtype is not None and not isinstance(array_dtype, np.dtype):
            try:
                array_dtype = np.dtype(array_dtype)
            except TypeError:
                if is_string_dtype(array_dtype):
                    array_dtype = np.dtype("object")
                else:
                    array_dtype = None

        # Non-Series objects implementing __cuda_array_interface__ or __array__
        # may not expose a valid ``dtype`` attribute themselves. Normalize
        # these before selecting from the supported dtypes so their represented
        # dtype is preserved.
        if array_dtype is None and not isinstance(
            array, (cudf.Series, pd.Series)
        ):
            if hasattr(array, "__cuda_array_interface__"):
                array = cp.asarray(array)
                array_dtype = array.dtype
            elif hasattr(array, "__array__"):
                array = np.asarray(array)
                array_dtype = array.dtype

    # Infer proper output dtype
    if array_dtype is not None:
        # Check for complex inputs before conversion when possible
        if array_dtype.kind == "c":
            raise ValueError("Complex data not supported")
        if dtype is None:
            dtype = array_dtype
        elif array_dtype not in dtype:
            # Convert to first provided dtype
            dtype = dtype[0]
        else:
            dtype = array_dtype
    elif dtype is not None:
        # No original dtype, use first provided dtype
        dtype = dtype[0]

    # Coerce `array` to numpy/cupy/scipy.sparse/cupyx.scipy.sparse values as
    # requested. For dataframe-like inputs also extract the index for later use.
    index = None
    if is_sparse:
        orig_sparse_array = array
        # Handle sparse inputs
        if isinstance(accept_sparse, str):
            accept_sparse = [accept_sparse]
        elif accept_sparse is True:
            # Only support formats cupyx.scipy.sparse supports
            accept_sparse = ["csr", "coo", "csc", "dia"]

        if not accept_sparse:
            padded_input = f" for {input_name}" if input_name else ""
            raise TypeError(
                f"Sparse data was passed{padded_input}, but dense data is required. "
                "Use '.toarray()' to convert to a dense array."
            )
        # Coerce to accepted format if needed
        if array.format not in accept_sparse:
            array = array.asformat(accept_sparse[0])
        if not accept_large_sparse:
            # Try to coerce to int32 indices, erroring otherwise
            array = _ensure_int32_sparse(array)
        elif (
            _requires_int64_sparse(array)
            and mem_type == "device"
            and not _CUPY_SUPPORTS_LARGE_SPARSE
        ):
            raise ValueError(
                "Sparse matrices with int64 indices require cupy >= 14.1.0"
            )

        # Validate dimensions and shape are as expected. We do this here
        # _before_ host/device conversion, since cupyx doesn't have a sparse
        # array type and thus only supports 2D inputs.
        _check_shape(
            array.shape,
            array_type=array_type,
            ensure_2d=ensure_2d,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
        )

        # Coerce to proper dtype and mem_type if needed
        if mem_type == "host" and not sp.issparse(array):
            # Coerce to host, then coerce dtype. We do this to save device
            # memory, and since scipy supports more dtypes.
            array = array.get()
            if dtype is not None and array.dtype != dtype:
                array = array.astype(dtype)
        else:
            # Otherwise coerce dtype, then mem_type if needed
            if dtype is not None and array.dtype != dtype:
                array = array.astype(dtype)
            if mem_type == "device" and not cp_sp.issparse(array):
                if array.ndim != 2:
                    raise ValueError(
                        "cupyx.scipy.sparse only supports 2D arrays"
                    )
                array = getattr(cp_sp, f"{array.format}_matrix")(array)

        # Copy if needed
        if copy and array is orig_sparse_array:
            array = array.copy()
    else:
        # Handle dense inputs
        if isinstance(array, (cudf.DataFrame, cudf.Series)):
            # Handle cudf inputs
            index = array.index
            if mem_type == "host":
                # XXX: cudf's to_numpy doesn't support conversions for all
                # dtypes. Roundtrip through object dtype when necessary.
                try:
                    array = array.to_numpy(dtype=dtype)
                except NotImplementedError:
                    array = array.to_numpy(dtype="object")
                array = np.asarray(array, dtype=dtype, order=order)
            else:
                array = cp.asarray(
                    array.to_cupy(copy=copy, dtype=dtype),
                    dtype=dtype,
                    order=order,
                )
        elif isinstance(array, (pd.DataFrame, pd.Series)):
            # Handle pandas inputs
            index = array.index
            # If cudf.pandas is enabled, copy=True always leads to a copy since
            # cudf.pandas may maintain dual host/device buffers. Otherwise
            # copy=True only leads to a copy here when outputting on host.
            array = array.to_numpy(
                dtype=dtype,
                copy=copy and (cudf.pandas.LOADED or mem_type != "device"),
            )
            if mem_type == "device":
                array = cp.asarray(array, dtype=dtype, order=order)
            elif (
                mem_type is None
                and cudf.pandas.LOADED
                and array.dtype.kind in "iufb"
            ):
                # We treat pandas objects with supported dtypes as device
                # memory when running under cudf.pandas. Note that the output
                # of `to_numpy` in cudf.pandas returns a proxy array that's
                # remains on device, so we're not paying a device<>host
                # roundtrip cost here.
                array = cp.asarray(array, dtype=dtype, order=order)
            else:
                array = np.asarray(array, dtype=dtype, order=order)
        elif hasattr(array, "__cuda_array_interface__"):
            # Handle device-backed array-like inputs
            if mem_type == "host":
                array = cp.asnumpy(array, order=order or "A")
                # Possible 2nd copy done on host for dtype enforcement
                array = np.asarray(array, dtype=dtype, order=order)
            else:
                # XXX: using cp.array for compat with cupy < 14
                array = cp.array(
                    array, dtype=dtype, order=order, copy=(copy or None)
                )
        else:
            # Handle all other inputs
            if mem_type == "device":
                # XXX: using cp.array for compat with cupy < 14
                array = cp.array(
                    array, dtype=dtype, order=order, copy=(copy or None)
                )
            else:
                array = np_asarray(
                    array, dtype=dtype, order=order, copy=(copy or None)
                )

        # XXX: order="A" isn't consistently handled by cupy or numpy. If a copy
        # was made, the output will definitely already be contiguous. If no
        # copy was already made though, we may need to make one to enforce
        # contiguity (here we default to F-contiguous, mirroring what _most_
        # code paths do with `order="A"`).
        if order == "A" and not (
            array.flags["F_CONTIGUOUS"] or array.flags["C_CONTIGUOUS"]
        ):
            array = (
                cp.asarray(array, order="F")
                if isinstance(array, cp.ndarray)
                else np.asarray(array, order="F")
            )

        # Validate dimensions and shape are as expected
        _check_shape(
            array.shape,
            array_type=array_type,
            ensure_2d=ensure_2d,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
        )

    # Check for complex inputs after conversion for cases when `dtype=None`
    if array.dtype.kind == "c":
        raise ValueError("Complex data not supported")

    # Validate data meets expected value requirements
    if ensure_all_finite:
        check_all_finite(
            array,
            allow_nan=ensure_all_finite == "allow-nan",
            input_name=input_name,
        )
    if ensure_non_negative:
        check_non_negative(array, input_name=input_name)

    # Process index if requested, then return
    if return_index:
        return array, _index_as_mem_type(index, mem_type)
    else:
        return array


def check_cudf(
    array,
    *,
    ensure_ndim=2,
    coerce_ndim=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    input_name=None,
):
    """Validate and coerce the input to a ``cudf.Series`` or ``cudf.DataFrame``.

    Parameters
    ----------
    array : array-like
        The input to validate.
    ensure_ndim : {1, 2, None}, default=2
        The number dimensions to enforce. Set to 1 to return a ``cudf.Series``,
        2 to return a cudf.DataFrame, or ``None`` to return either.
    coerce_ndim : bool or "warn", default=False
        Whether to allow coercing between 1d and 2d inputs. May also set to
        "warn" to warn the user to reshape their input.
    ensure_min_samples : int, default=1
        A minimum number of samples to require. Set to 0 for no minimum.
    ensure_min_features : int, default=1
        A minimum number of features to require for 2D inputs. Set to 0 for no
        minimum.
    input_name : str or None, default=None
        The input parameter name to use in error messages.

    Returns
    -------
    array : cudf.Series or cudf.DataFrame
        The converted and validated data.
    """
    if ensure_ndim not in (1, 2, None):
        raise ValueError(f"Unsupported {ensure_ndim=!r}")
    if coerce_ndim not in (True, False, "warn"):
        raise ValueError(f"Unsupported {coerce_ndim=!r}")
    if ensure_min_features > 1 and ensure_ndim != 2:
        raise ValueError(f"{ensure_min_features=!r} requires ensure_ndim=2")

    if cp_sp.issparse(array) or sp.issparse(array):
        padded_input = f" for {input_name}" if input_name else ""
        raise TypeError(
            f"Sparse data was passed{padded_input}, but dense data is required. "
            "Use '.toarray()' to convert to a dense array."
        )

    array_type = type(array)

    # Coerce input to a cudf type.
    # XXX: cudf currently doesn't support float16, any float16 input is
    # automatically upcast here to float32.
    # XXX: hardcode `nan_as_null=True` (cudf's default) so the behavior
    # doesn't switch when cudf.pandas is active.
    if not isinstance(array, (cudf.DataFrame, cudf.Series)):
        array_dtype_kind = None
        if isinstance(array, pd.Series):
            if array.dtype == "float16":
                array = array.astype("float32")
        elif isinstance(array, pd.DataFrame):
            f16_cols = array.select_dtypes("float16").columns.tolist()
            if f16_cols:
                array = array.astype({c: "float32" for c in f16_cols})
        else:
            # Normalize to numpy or cupy array with minimal copying
            if hasattr(array, "__cuda_array_interface__"):
                array = cp.asarray(array)
            elif hasattr(array, "__array__") or hasattr(
                array, "__array_interface__"
            ):
                array = np.asarray(array)
            elif not isinstance(array, np.ndarray):
                array = np.asarray(array, dtype=object)

            if array.dtype.kind == "c":
                raise ValueError("Complex data not supported")

            if array.dtype == "float16":
                array = array.astype("float32")
            array_dtype_kind = array.dtype.kind

        array_shape = array.shape
        cls = cudf.DataFrame if array.ndim == 2 else cudf.Series
        if array_dtype_kind == "O":
            # For object dtype inputs, coerce back to list (cheap) to rely on
            # cudf's per-column dtype inference. On failure raise an error
            # compatible with what sklearn's `check_dtype_object` expects.
            array = array.tolist()

        # Try to coerce to cudf, raising a nicer error message on failure.
        try:
            array = cls(array, nan_as_null=True)
        except Exception as exc:
            # XXX: cudf throws a number of different errors when handed a
            # container with bytes inputs. Here we try to detect all of them,
            # falling back to a generic message about mixed object dtypes.
            needle = None
            if isinstance(exc, ValueError):
                needle = "type_id"
            elif isinstance(
                exc, (NotImplementedError, cudf.errors.MixedTypeError)
            ):
                needle = "bytes"
            if needle and needle in str(exc).lower():
                raise TypeError(
                    f"{input_name or 'Input'} with bytes dtype is not supported. "
                    "Try converting to strings first."
                ) from exc
            raise TypeError(
                f"An object dtype {input_name or 'input'} argument must be "
                "composed of strings, numbers, booleans, or nulls."
            ) from exc
    else:
        array_shape = array.shape

    # Validate shape and coerce dimensionality
    _check_shape(
        array_shape,
        ensure_2d=(ensure_ndim == 2 and coerce_ndim is False),
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        array_type=array_type,
    )
    if ensure_ndim == 1:
        # Warn/error appropriately for 2D inputs
        if array.ndim == 2:
            if not coerce_ndim or array.shape[1] != 1:
                raise ValueError(
                    f"{input_name or 'Input'} should be a 1d array, got an array "
                    f"of shape {array.shape} instead."
                )
            if coerce_ndim == "warn":
                name = input_name or "input"
                warnings.warn(
                    f"A column-vector {name} was passed when a 1d array was "
                    f"expected. Please change the shape of {name} to "
                    f"(n_samples,), for example using ravel().",
                    DataConversionWarning,
                )
            array = array.iloc[:, 0]
    elif ensure_ndim == 2 and array.ndim == 1:
        # coerce_ndim == False is handled above in _check_shape
        if coerce_ndim == "warn":
            name = input_name or "input"
            warnings.warn(
                f"A 1d array {name} was passed when a 2d array was "
                f"expected. Please change the shape of {name} to "
                f"(n_samples, 1), for example using reshape(-1, 1).",
                DataConversionWarning,
            )
        array = array.to_frame()

    return array


# Returns status in a bitfield:
# 0b001: contains NaN
# 0b010: contains +/-inf
# 0b100: contains real (non-integral) values
_cupy_any_inf_or_nan_or_real = cp.ReductionKernel(
    "T x",
    "uint8 out",
    (
        "((unsigned char)(ceil(x) != x)) << 2 "
        "| ((unsigned char)isinf(x)) << 1 "
        "| ((unsigned char)isnan(x))"
    ),
    "a | b",
    "out = a",
    "0",
    "any_inf_or_nan_or_real",
)


def check_classification_targets(y):
    """Check if `y` is composed of valid class labels.

    Catches NaN, infinity, and non-integral inputs.

    Parameters
    ----------
    y : cupy.ndarray
        The ``y`` input to check.
    """
    if y.dtype.kind == "f":
        status = _cupy_any_inf_or_nan_or_real(y)
        if status & 0b001:
            raise ValueError("Input y contains NaN.")
        elif status & 0b010:
            raise ValueError(
                f"Input y contains infinity or a value too large for {y.dtype!r}."
            )
        elif status & 0b100:
            raise ValueError(
                "Unknown label type: continuous. Maybe you are trying to fit a "
                "classifier, which expects discrete classes on a regression target "
                "with continuous values."
            )


def check_y(
    y,
    *,
    dtype=None,
    mem_type="device",
    order="A",
    accept_multi_output=False,
    ensure_discrete_classes=True,
    return_classes=False,
    return_index=False,
):
    """Validate and coerce ``y`` to a supported type.

    Parameters
    ----------
    y : array-like
        The array-like input to validate.
    dtype : None, dtype, list[dtype], default=None
        The dtype(s) to support. By default no dtype enforcement is performed;
        for classifiers the output will be a suitable integral type, otherwise
        the input dtype will be used. Pass a dtype or a list of supported
        dtypes to enforce a dtype for the output. If the input doesn't have a
        supported dtype, it will be converted to the first listed dtype.
    mem_type : {'device', 'host'} or None, default='device'
        The memory type use for the output. If 'device', the output will be a
        ``cupy.ndarray``. If 'host', the output will be a ``numpy.ndarray``. If
        ``None``, the output will have the same memory type as the input (i.e.
        device if already on device, host otherwise).
    order : {'F', 'C', 'A', None}, default='A'
        The order and contiguity to enforce for dense outputs. Use 'F' for
        F-contiguous outputs, 'C' for C-contiguous outputs, 'A' for either F or
        C contiguous, or `None` for no contiguity requirements (may be
        non-contiguous!).
    accept_multi_output : bool, default=False
        Whether multi-output y is accepted. By default only 1D inputs (or 2D
        inputs with a single column) are accepted. Set to True to accept
        multi-column inputs as well.
    ensure_discrete_classes : bool, default=True
        Whether to ensure class labels are discrete, non-continuous values.
    return_classes : bool, np.ndarray, or list[numpy.ndarray], default=False
        Set to True to also label encode ``y`` and return the ``classes``.
        Alternatively may pass a numpy array (or a list of arrays in the
        multi-output case) to explicitly specify the ``classes`` to use for
        label encoding.
    return_index : bool, default=False
        Whether to return the index of ``y`` (if a dataframe-like value).
        This is useful for functions that need to return an output with a
        dataframe index aligned with the input.

    Returns
    -------
    y : cupy.ndarray or numpy.ndarray
        The converted and validated array.
    classes : numpy.ndarray or list[numpy.ndarray]
        The collected classes for a classifier input. Only returned if
        ``return_classes`` is not ``False``.
    index : pandas.Index, cudf.Index, or None
        The index of the input if a dataframe-like, or None if no index. The
        index will be converted to match ``mem_type``. Only returned if
        ``return_index=True``.
    """
    if y is None:
        raise ValueError(
            "This estimator requires y to be passed, but the target y is None"
        )

    # Normalize `dtype` arg
    if dtype is not None:
        if not isinstance(dtype, (list, tuple)):
            dtype = [dtype]
        dtype = [np.dtype(i) for i in dtype]

    # Extract the index from `y` (if available)
    if isinstance(y, (pd.DataFrame, pd.Series, cudf.DataFrame, cudf.Series)):
        index = y.index
    else:
        index = None

    # Coerce `y` to a supported array type
    if return_classes is not False:
        # cudf may coerce the dtype, store the original so we can cast back later
        input_dtype = y.dtype if isinstance(y, np.ndarray) else None

        # No cuda container supports all dtypes. Here we coerce to cupy when
        # possible, falling back to cudf Series/DataFrame otherwise.
        if not isinstance(y, (cudf.DataFrame, cudf.Series)):
            y = check_array(
                y,
                mem_type=None,
                ensure_2d=False,
                ensure_min_samples=0,
                ensure_all_finite=False,
                input_name="y",
            )
            # If no original dtype found on input, use the coerced one instead
            if input_dtype is None:
                input_dtype = y.dtype
            if mem_type is None:
                mem_type = "host" if isinstance(y, np.ndarray) else "device"
            if y.dtype.kind in "iufb" and return_classes is True:
                y = cp.asarray(y)
            elif (
                y.dtype == "object"
                and y.size
                and not isinstance(y.flat[0], str)
            ):
                raise ValueError(
                    "Unknown label type: unknown. Maybe you are trying to fit a "
                    "classifier, which expects discrete classes on a regression target "
                    "with continuous values."
                )
            else:
                y = (cudf.DataFrame if y.ndim == 2 else cudf.Series)(
                    y, dtype=(np.dtype("O") if y.dtype.kind in "U" else None)
                )
        elif mem_type is None:
            mem_type = "device"
    else:
        y = check_array(
            y,
            dtype=dtype,
            mem_type=mem_type,
            order=order,
            ensure_2d=False,
            ensure_min_samples=0,
            input_name="y",
        )
        mem_type = "device" if isinstance(y, cp.ndarray) else "host"

    # Warn/error appropriately for 2D inputs
    if y.ndim == 2:
        if y.shape[1] == 1 and (
            return_classes is not False or not accept_multi_output
        ):
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,), for example "
                "using ravel().",
                DataConversionWarning,
            )
            if return_classes is not False:
                y = (
                    y.iloc[:, 0]
                    if isinstance(y, cudf.DataFrame)
                    else y.ravel()
                )
        elif not accept_multi_output:
            raise ValueError(
                f"y should be a 1d array, got an array of shape {y.shape} instead."
            )

    if return_index:
        index = _index_as_mem_type(index, mem_type)

    if return_classes is False:
        return (y, index) if return_index else y

    # For classifiers, we label encode y and return the integral labels as well
    # as the classes.
    def _encode(y, classes=None):
        """Encode `y` to codes and classes"""
        if ensure_discrete_classes:
            check_classification_targets(y)
        if isinstance(y, cudf.Series):
            y = y.astype("category")
            if classes is None:
                classes = y.cat.categories
                # XXX: cudf's to_numpy doesn't support conversions for all
                # dtypes. Roundtrip through object dtype when necessary.
                try:
                    classes = classes.to_numpy(dtype=input_dtype)
                except NotImplementedError:
                    classes = classes.to_numpy(dtype="object")
                # cudf will sometimes translate non-numeric dtypes. Coerce back to
                # the input dtype if the input was originally a numpy array.
                if input_dtype is not None:
                    classes = classes.astype(input_dtype, copy=False)
            else:
                # Encode using specified classes, erroring if any unknown
                # classes found
                y_orig = y
                y = y.cat.set_categories(classes)
                if y.has_nulls or y.hasnans:
                    new_cats = y_orig.cat.categories
                    unknown = new_cats[~new_cats.isin(classes)].to_numpy(
                        dtype="object"
                    )
                    raise ValueError(
                        f"The target label(s) {unknown!s} in y do not exist in the "
                        f"initial classes {classes!s}"
                    )
            codes = cp.asarray(y.cat.codes)
        else:
            assert classes is None  # we should always use cudf in this case
            classes, codes = cp.unique(y, return_inverse=True)
            classes = classes.get()
        return codes, classes

    # Return C order if C requested, otherwise F.
    if order != "C":
        order = "F"

    if y.ndim == 1:
        if return_classes is True:
            provided_classes = None
        elif not isinstance(return_classes, np.ndarray):
            raise ValueError("Expected `return_classes` to be a numpy array")
        else:
            provided_classes = return_classes

        y, classes = _encode(y, provided_classes)
        if dtype is not None and y.dtype not in dtype:
            y = y.astype(dtype[0])
    else:
        getter = y.iloc if isinstance(y, cudf.DataFrame) else y

        if return_classes is True:
            provided_classes = [None] * y.shape[1]
        elif not (
            isinstance(return_classes, list)
            and len(return_classes) == y.shape[1]
            and all(isinstance(a, np.ndarray) for a in return_classes)
        ):
            raise ValueError(
                f"Expected `return_classes` to be a list of {y.shape[1]} numpy arrays"
            )
        else:
            provided_classes = return_classes

        encoded_cols, classes = zip(
            *(
                _encode(getter[:, i], provided_classes[i])
                for i in range(y.shape[1])
            )
        )
        classes = list(classes)
        # Infer output dtype
        out_dtype = cp.result_type(*(c.dtype for c in encoded_cols))
        if dtype is not None and out_dtype not in dtype:
            dtype = dtype[0]
        else:
            dtype = out_dtype
        y = cp.empty(shape=y.shape, dtype=dtype, order=order)
        for i, col in enumerate(encoded_cols):
            y[:, i] = col

    if mem_type == "host":
        # convert back to host if needed
        y = y.get(order=order)

    if return_index:
        return y, classes, index
    return y, classes


def check_sample_weight(
    sample_weight,
    *,
    dtype=None,
    mem_type="device",
    order="A",
    ensure_non_negative=False,
):
    """Validate and coerce ``sample_weight`` to a supported type.

    Parameters
    ----------
    sample_weight : array-like, scalar, or None
        The ``sample_weight`` input to validate.
    dtype : None, dtype, list[dtype], default=None
        The dtype(s) to support. By default no dtype validation is performed.
        Pass a dtype or a list of supported dtypes to enforce a dtype for the
        output. If the input doesn't have a supported dtype, it will be
        converted to the first listed dtype.
    mem_type : {'device', 'host'} or None, default='device'
        The memory type use for the output. If 'device', the output will be a
        ``cupy.ndarray``. If 'host', the output will be a ``numpy.ndarray``. If
        ``None``, the output will have the same memory type as the input (i.e.
        device if already on device, host otherwise).
    order : {'F', 'C', 'A', None}, default='A'
        The order and contiguity to enforce for dense outputs. Use 'F' for
        F-contiguous outputs, 'C' for C-contiguous outputs, 'A' for either F or
        C contiguous, or `None` for no contiguity requirements (may be
        non-contiguous!).
    ensure_non_negative : bool, default=False
        If True, an error will be raised if negative values are found in the
        input. By default ``check_non_negative`` is skipped.

    Returns
    -------
    sample_weight : cupy.ndarray, numpy.ndarray, or None
        The converted and validated weights.
    """
    if sample_weight is None:
        return None

    all_zero_msg = "Sample weights must contain at least one non-zero number."

    if np.isscalar(sample_weight):
        if sample_weight == 0:
            raise ValueError(all_zero_msg)
        elif ensure_non_negative and sample_weight < 0:
            raise ValueError("Negative values in data passed to sample_weight")
        elif np.isnan(sample_weight):
            raise ValueError("Input sample_weight contains NaN")
        elif np.isinf(sample_weight):
            raise ValueError("Input sample_weight contains infinity")
        else:
            # A uniform sample_weight is the same as unweighted
            return None

    sample_weight = check_array(
        sample_weight,
        dtype=dtype,
        mem_type=mem_type,
        order=order,
        ensure_2d=False,
        ensure_min_samples=0,
        ensure_non_negative=ensure_non_negative,
        input_name="sample_weight",
    )
    if sample_weight.ndim != 1:
        raise ValueError(
            f"Sample weights must be 1D array or scalar, got "
            f"{sample_weight.ndim}D array."
        )

    if sample_weight.size and (sample_weight == 0).all():
        raise ValueError(all_zero_msg)
    return sample_weight


def check_inputs(
    estimator,
    X,
    y=...,
    sample_weight=...,
    *,
    accept_sparse=False,
    accept_large_sparse=False,
    dtype=None,
    y_dtype=...,
    sample_weight_dtype=...,
    mem_type="device",
    order="A",
    copy=False,
    ensure_all_finite=True,
    ensure_non_negative=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    accept_multi_output=False,
    return_classes=False,
    return_index=False,
    reset=False,
):
    """Validate and coerce common inputs to an estimator method.

    This plumbs together several common checks. For a method with ``X``, ``y``,
    and ``sample_weight``, it's roughly equivalent to:

    ```
    check_features(estimator, X, reset=reset)
    X = check_array(X, input_name="X", ...)
    y = check_y(y, ...)
    sample_weight = check_sample_weight(sample_weight, ...)
    check_consistent_length(X, y, sample_weight)
    ```

    If this pattern doesn't work for an estimator, you can call always call
    some of the individual checks directly.

    Parameters
    ----------
    estimator : Base
        The estimator to check.
    X : array-like
        The ``X`` input.
    y : array-like, default=...
        The ``y`` input. May be omitted.
    sample_weight : array-like, scalar, or None
        The ``sample_weight`` input. May be omitted.
    accept_sparse : bool, str, list[str], default=False
        The sparse matrix format(s) to support. If the input is sparse
        but not in a supported format, it will be converted to the first
        listed format. Pass True to support any input format. The default
        of False will raise an error on sparse inputs.
    accept_large_sparse : bool, default=False
        Whether large (int64) indices are supported for sparse containers with
        CSR/CSC/COO/BSR formats. If not supported, an appropriate error will be
        raised if the sparse indices aren't int32.
    dtype : None, dtype, list[dtype], default=None
        The dtype(s) to support for X. By default no dtype validation is performed.
        Pass a dtype or a list of supported dtypes to enforce a dtype for the
        output. If the input doesn't have a supported dtype, it will be
        converted to the first listed dtype.
    y_dtype : None, dtype, list[dtype], default=...
        The dtype(s) to support for y. If not specified, defaults to
        the output dtype of ``X``.
    sample_weight_dtype : None, dtype, list[dtype], default=...
        The dtype(s) to support for sample_weight. If not specified, defaults
        to the output dtype of ``X``.
    mem_type : {'device', 'host'} or None, default='device'
        The memory type use for the output. If 'device', the output will be a
        ``cupy.ndarray`` if dense, or a ``cupyx.scipy.sparse.spmatrix`` if
        sparse. If 'host', the output will be a ``numpy.ndarray`` if dense, or
        a ``scipy.sparse.spmatrix`` if sparse. If ``None``, the output will
        have the same memory type as the input (i.e. device if already on
        device, host otherwise).
    order : {'F', 'C', 'A', None}, default='A'
        The order and contiguity to enforce for dense outputs. Use 'F' for
        F-contiguous outputs, 'C' for C-contiguous outputs, 'A' for either F or
        C contiguous, or `None` for no contiguity requirements (may be
        non-contiguous!).
    copy : bool, default=False
        Set to ``True`` to ensure that X is copied (allowing the output to
        be mutated without worry). Note that the default of ``False`` doesn't
        guarantee no copy is made, it only allows for zero-copy when possible.
    ensure_all_finite : bool or 'allow-nan', default=True
        If True, an error will be raised if non-finite values are found in X.
        If 'allow-nan', an error will be raised if infinite values are
        found (but not for NaN). If False then ``check_all_finite`` is skipped.
    ensure_non_negative : bool, default=False
        If True, an error will be raised if negative values are found in X. By
        default ``check_non_negative`` is skipped.
    ensure_min_samples : int, default=1
        A minimum number of samples to require. Set to 0 for no minimum.
    ensure_min_features : int, default=1
        A minimum number of features to require for 2D inputs. Set to 0 for no
        minimum.
    accept_multi_output : bool, default=False
        Whether multi-output y is accepted. By default only 1D inputs (or 2D
        inputs with a single column) are accepted. Set to True to accept
        multi-column inputs as well.
    return_classes : bool, np.ndarray, or list[numpy.ndarray], default=False
        Set to True to also label encode ``y`` and return the ``classes``.
        Alternatively may pass a numpy array (or a list of arrays in the
        multi-output case) to explicitly specify the ``classes`` to use for
        label encoding.
    return_index : bool, default=False
        Whether to return the index of ``X`` (if a dataframe-like value).
        This is useful for functions that need to return an output with a
        dataframe index aligned with the input.
    reset : bool, default=False
        If True, ``n_features_in_`` and ``feature_names_in_`` are set on
        ``estimator`` to match ``X``. Otherwise ``X`` is checked to match the
        existing ``n_features_in_`` and ``feature_names_in_``. ``reset=True``
        should be used for fit-like methods, and False otherwise.

    Returns
    -------
    X : dense or sparse array
        The converted and validated array. Depending on input and parameters,
        will be one of ``cupy.ndarray``, ``numpy.ndarray``,
        ``cupyx.scipy.sparse.spmatrix``, or ``scipy.sparse.spmatrix``.
    y : cupy.ndarray or numpy.ndarry
        The converted and validated array. Omitted if no ``y`` provided.
    sample_weight : cupy.ndarray, numpy.ndarray, or None
        The converted and validated weights. Omitted if no ``sample_weight``
        provided.
    classes : numpy.ndarray or list[numpy.ndarray]
        The collected classes for a classifier input. Only returned if
        ``return_classes`` is not ``False``.
    index : pandas.Index, cudf.Index, or None
        The index of the input if a dataframe-like, or None if no index. The
        index will be converted to match ``mem_type``. Only returned if
        ``return_index=True``.
    """
    check_features(estimator, X, reset=reset)

    # Validate X
    X = check_array(
        X,
        accept_sparse=accept_sparse,
        accept_large_sparse=accept_large_sparse,
        dtype=dtype,
        mem_type=mem_type,
        order=order,
        copy=copy,
        ensure_all_finite=ensure_all_finite,
        ensure_non_negative=ensure_non_negative,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        return_index=return_index,
        input_name="X",
    )
    if return_index:
        X, index = X
    else:
        index = None
    out = [X]

    # Validate y
    classes = None
    if y is not ...:
        if y_dtype is ...:
            y_dtype = X.dtype
        y = check_y(
            y,
            dtype=y_dtype,
            mem_type=mem_type,
            order=order,
            accept_multi_output=accept_multi_output,
            return_classes=return_classes,
        )
        if return_classes is not False:
            y, classes = y
        out.append(y)

    # Validate sample_weight
    if sample_weight is not ...:
        if sample_weight_dtype is ...:
            sample_weight_dtype = X.dtype
        sample_weight = check_sample_weight(
            sample_weight,
            dtype=sample_weight_dtype,
            mem_type=mem_type,
            order=order,
        )
        out.append(sample_weight)

    check_consistent_length(*out)

    if return_classes is not False:
        out.append(classes)
    if return_index:
        out.append(index)

    return out[0] if len(out) == 1 else tuple(out)
