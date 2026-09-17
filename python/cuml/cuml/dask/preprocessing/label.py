# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp

from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.input_utils import _extract_partitions
from cuml.preprocessing.label import LabelBinarizer as LB


class LabelBinarizer(BaseEstimator):
    """
    A distributed version of LabelBinarizer for one-hot encoding
    a collection of labels.

    Parameters
    ----------
    client : dask.distributed.Client, optional
        Dask client to use
    **kwargs : dict
        Additional arguments passed to the underlying single-GPU LabelBinarizer

    Examples
    --------

    Create an array with labels and dummy encode them

    .. code-block:: python

        >>> import cupy as cp
        >>> import cupyx
        >>> from cuml.dask.preprocessing import LabelBinarizer

        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client
        >>> import dask

        >>> cluster = LocalCUDACluster()
        >>> client = Client(cluster)

        >>> labels = cp.asarray([0, 5, 10, 7, 2, 4, 1, 0, 0, 4, 3, 2, 1],
        ...                     dtype=cp.int32)
        >>> labels = dask.array.from_array(labels)

        >>> lb = LabelBinarizer()
        >>> encoded = lb.fit_transform(labels)
        >>> print(encoded.compute())
        [[1 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 0 0]
        [0 0 0 0 0 0 0 1]
        [0 0 0 0 0 0 1 0]
        [0 0 1 0 0 0 0 0]
        [0 0 0 0 1 0 0 0]
        [0 1 0 0 0 0 0 0]
        [1 0 0 0 0 0 0 0]
        [1 0 0 0 0 0 0 0]
        [0 0 0 0 1 0 0 0]
        [0 0 0 1 0 0 0 0]
        [0 0 1 0 0 0 0 0]
        [0 1 0 0 0 0 0 0]]
        >>> decoded = lb.inverse_transform(encoded)
        >>> print(decoded.compute())
        [ 0  5 10  7  2  4  1  0  0  4  3  2  1]
        >>> client.close()
        >>> cluster.close()

    """

    def __init__(self, *, client=None, **kwargs):
        super().__init__(client=client, **kwargs)
        # Sparse output will be added once sparse CuPy arrays are supported
        # by Dask.Array: https://github.com/NVIDIA/cuml/issues/1665
        if (
            "sparse_output" in self.kwargs
            and self.kwargs["sparse_output"] is True
        ):
            raise ValueError(
                "Sparse output not yet supported in distributed mode"
            )

    @staticmethod
    def _func_create_model(**kwargs):
        return LB(**kwargs)

    @staticmethod
    def _func_xform(y, *, model):
        xform_in = cp.asarray(y, dtype=y.dtype)
        return model.transform(xform_in)

    @staticmethod
    def _func_inv_xform(y, *, model, threshold):
        y = cp.asarray(y, dtype=y.dtype)
        return model.inverse_transform(y, threshold=threshold)

    def fit(self, y):
        """Fit label binarizer

        Parameters
        ----------
        y : Dask.Array of shape [n_samples,] or [n_samples, n_classes]
            chunked by row.
            Target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """

        # Take the unique classes and broadcast them all around the cluster.
        futures = self.client.sync(_extract_partitions, y)

        unique = [self.client.submit(cp.unique, f) for w, f in futures]

        classes = self.client.compute(unique, True)
        classes = cp.unique(cp.stack(classes, axis=0))

        self._set_internal_model(LB(**self.kwargs).fit(classes))

        return self

    def fit_transform(self, y):
        """
        Fit the label encoder and return transformed labels

        Parameters
        ----------
        y : Dask.Array of shape [n_samples,] or [n_samples, n_classes]
            target values. The 2-d matrix should only contain 0 and 1,
            represents multilabel classification.

        Returns
        -------

        arr : Dask.Array backed by CuPy arrays containing encoded labels
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """
        Transform and return encoded labels

        Parameters
        ----------
        y : Dask.Array of shape [n_samples,] or [n_samples, n_classes]

        Returns
        -------

        arr : Dask.Array backed by CuPy arrays containing encoded labels
        """
        new_axis = 1 if y.ndim == 1 else None
        return y.map_blocks(
            LabelBinarizer._func_xform,
            model=self._get_internal_model(),
            dtype=cp.float32,
            new_axis=new_axis,
        )

    def inverse_transform(self, y, threshold=None):
        """
        Invert a set of encoded labels back to original labels

        Parameters
        ----------

        y : Dask.Array of shape [n_samples, n_classes] containing encoded
            labels

        threshold : float This value is currently ignored

        Returns
        -------

        arr : Dask.Array backed by CuPy arrays containing original labels
        """
        return y.map_blocks(
            LabelBinarizer._func_inv_xform,
            model=self._get_internal_model(),
            dtype=y.dtype,
            threshold=threshold,
            drop_axis=1,
        )
