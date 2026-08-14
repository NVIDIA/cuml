# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import cupyx.scipy.sparse as sp
import numpy as np

from cuml.common.doc_utils import generate_docstring
from cuml.internals.base import Base
from cuml.internals.mixins import SparseInputTagMixin
from cuml.internals.outputs import ReflectedAttr, mlfunc
from cuml.internals.validation import (
    check_inputs,
    check_is_fitted,
    check_random_seed,
)


def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
    """
    Find a 'safe' number of components to randomly project to.

    The Johnson–Lindenstrauss lemma states that high-dimensional data can be
    embedded into lower dimension while preserving the distances.

    This function finds the minimum number of components to guarantee that
    the embedding is inside the eps error tolerance.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    eps : float in (0,1) (default = 0.1)
        Maximum distortion rate as defined by the Johnson-Lindenstrauss lemma.

    Returns
    -------
    n_components : int
        The minimal number of components to guarantee with good probability
        an eps-embedding with n_samples.
    """
    import sklearn.random_projection

    return sklearn.random_projection.johnson_lindenstrauss_min_dim(
        n_samples, eps=eps
    )


class _BaseRandomProjection(SparseInputTagMixin, Base):
    """Base class for RandomProjection estimators."""

    components_ = ReflectedAttr()

    def __init__(
        self,
        n_components="auto",
        *,
        eps=0.1,
        random_state=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(output_type=output_type, verbose=verbose)
        self.n_components = n_components
        self.eps = eps
        self.random_state = random_state

    @classmethod
    def _get_param_names(cls):
        return [
            "n_components",
            "eps",
            "random_state",
            *super()._get_param_names(),
        ]

    def _gen_random_matrix(self, n_components, n_features, dtype):
        raise NotImplementedError

    @generate_docstring()
    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        """Generate a random projection matrix."""
        # Use `mem_type=None` & `order=None` to minimize copies or transfers. We
        # don't need to access the data here, just ensure it's valid and get
        # the shape/dtype
        X = check_inputs(
            self,
            X,
            dtype=("float32", "float64"),
            mem_type=None,
            order=None,
            accept_sparse=True,
            accept_large_sparse=True,
            reset=True,
        )
        n_samples, n_features = X.shape

        if self.n_components == "auto":
            self.n_components_ = johnson_lindenstrauss_min_dim(
                n_samples=n_samples, eps=self.eps
            )
            if self.n_components_ > n_features:
                raise ValueError(
                    f"eps={self.eps} and {n_samples=} lead to a target dimension of "
                    f"{self.n_components_} which is larger than the original space with "
                    f"{n_features=}"
                )
        else:
            self.n_components_ = self.n_components

        if self.n_components_ <= 0:
            raise ValueError(
                f"n_components must be strictly positive, got {self.n_components_}"
            )

        self.components_ = self._gen_random_matrix(
            self.n_components_, n_features, X.dtype
        )

        return self

    @generate_docstring()
    @mlfunc(preserve_index=True)
    def transform(self, X):
        """Project the data by taking the matrix product with the random matrix."""
        check_is_fitted(self)
        X = check_inputs(
            self,
            X,
            dtype=("float32", "float64"),
            accept_sparse=("csr", "csc"),
            accept_large_sparse=True,
        )

        # Compute the output
        out = X @ self.components_.T

        # Coerce to correct dtype if needed (sparse matrices astype doesn't
        # support copy=False, so we need to use the more verbose version here).
        if out.dtype != self.components_.dtype:
            out = out.astype(self.components_.dtype)

        if sp.issparse(out):
            if not getattr(self, "dense_output", False):
                # Sparse output
                return out
            out = out.toarray()

        return out

    @generate_docstring()
    @mlfunc(preserve_index=True)
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)


class GaussianRandomProjection(_BaseRandomProjection):
    """Reduce dimensionality through Gaussian random projection.

    The components of the random matrix are drawn from N(0, 1 / n_components).

    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    eps : float, default=0.1
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when `n_components` is set to
        'auto'. The value should be strictly positive.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the
        projection matrix at fit time.

    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    n_components_ : int
        Concrete number of components computed when n_components="auto".

    components_ : array of shape (n_components, n_features)
        Random matrix used for the projection.

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from cuml.random_projection import GaussianRandomProjection
    >>> from cuml.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, n_features=1000, random_state=42)
    >>> model = GaussianRandomProjection(n_components=50, random_state=42)
    >>> X_new = model.fit_transform(X)
    >>> X_new.shape
    (200, 50)

    Notes
    -----
    Inspired by Scikit-learn's implementation:
    https://scikit-learn.org/stable/modules/random_projection.html

    Currently passing a sparse array to `transform` may result in close (but
    not exactly identical) results due to https://github.com/cupy/cupy/issues/9323.
    """

    def _gen_random_matrix(self, n_components, n_features, dtype):
        seed = check_random_seed(self.random_state)
        rng = cp.random.RandomState(seed)
        return rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(n_components),
            size=(n_components, n_features),
        ).astype(dtype, copy=False)


class SparseRandomProjection(_BaseRandomProjection):
    """Reduce dimensionality through sparse random projection.

    Sparse random matrix is an alternative to dense random projection matrix
    that guarantees similar embedding quality while being much more memory
    efficient and allowing faster computation of the projected data.

    If we note `s = 1 / density` the components of the random matrix are
    drawn from:

    .. code-block:: text

      -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
       0                              with probability 1 - 1 / s
      +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    density : float or 'auto', default='auto'
        Ratio in the range (0, 1] of non-zero component in the random
        projection matrix.

        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).

    eps : float, default=0.1
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'. This value should be strictly positive.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    dense_output : bool, default=False
        If True, ensure that the output of the random projection is a dense
        array even if the input and random projection matrix are both sparse.
        If False, the projected data uses a sparse representation if the input
        is sparse.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the
        projection matrix at fit time.

    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    n_components_ : int
        Concrete number of components computed when n_components="auto".

    components_ : sparse matrix of shape (n_components, n_features)
        Random matrix used for the projection.

    density_ : float in range 0.0 - 1.0
        Concrete density computed from when density = "auto".

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from cuml.random_projection import SparseRandomProjection
    >>> from cuml.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, n_features=1000, random_state=42)
    >>> model = SparseRandomProjection(n_components=50, random_state=42)
    >>> X_new = model.fit_transform(X)
    >>> X_new.shape
    (200, 50)

    Notes
    -----
    Inspired by Scikit-learn's implementation:
    https://scikit-learn.org/stable/modules/random_projection.html

    Currently passing a dense array to `transform` may result in close (but
    not exactly identical) results due to https://github.com/cupy/cupy/issues/9323.
    """

    def __init__(
        self,
        n_components="auto",
        *,
        density="auto",
        eps=0.1,
        dense_output=False,
        random_state=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(
            n_components=n_components,
            eps=eps,
            random_state=random_state,
            verbose=verbose,
            output_type=output_type,
        )
        self.density = density
        self.dense_output = dense_output

    @classmethod
    def _get_param_names(cls):
        return [
            "density",
            "dense_output",
            *super()._get_param_names(),
        ]

    def _gen_random_matrix(self, n_components, n_features, dtype):
        if self.density == "auto":
            density = 1 / np.sqrt(n_features)
        elif self.density <= 0 or self.density > 1:
            raise ValueError(f"Expected 0 < density <= 1, got {self.density}")
        else:
            density = self.density

        self.density_ = density

        seed = check_random_seed(self.random_state)
        rng = cp.random.default_rng(seed)

        if density == 1:
            # Completely dense, generate a dense matrix
            components = (
                rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
            )
            return (1 / np.sqrt(n_components) * components).astype(
                dtype, copy=False
            )

        k = int(density * n_features * n_components)
        # Index generation performed on host. cupy's implementation of choice
        # with replace=False isn't currently memory efficient. See
        # https://github.com/cupy/cupy/issues/9320. This operation runs fine on
        # host; it's necessarily important to move this to run on GPU anyway.
        ind = cp.asarray(
            np.random.default_rng(seed).choice(
                int(n_features * n_components), k, replace=False
            )
        )
        i = ind // n_features
        j = ind - i * n_features

        data = rng.binomial(1, 0.5, size=k)
        data *= 2
        data -= 1
        factor = np.sqrt(1 / density) / np.sqrt(n_components)
        data = cp.multiply(data, factor, dtype=dtype)

        return sp.coo_matrix(
            (data, (i, j)), shape=(n_components, n_features)
        ).asformat("csr")
