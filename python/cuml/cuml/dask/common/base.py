# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from collections.abc import Iterable
from functools import wraps

import cudf.comm.serialize  # noqa: F401
import cupy as cp
import dask
import dask_cudf
import numpy as np
from dask_cudf import DataFrame as dcDataFrame
from dask_cudf import Series as dcSeries
from distributed.client import Future
from raft_dask.common.comms import Comms
from toolz import first

from cuml.dask.common import parts_to_ranks
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.utils import get_client, wait_and_raise_from_futures
from cuml.internals.base import Base


class BaseEstimator:
    def __init__(self, *, client=None, verbose=False, **kwargs):
        """
        Constructor for distributed estimators.
        """
        self.client = get_client(client)

        # set client verbosity
        self.verbose = verbose

        # kwargs transmits the verbosity level to workers
        kwargs["verbose"] = verbose
        self.kwargs = kwargs

        self.internal_model = None

    def __getstate__(self):
        internal_model = self._get_internal_model()
        if internal_model:
            internal_model = internal_model.result()
        state = {
            "verbose": self.verbose,
            "kwargs": self.kwargs,
            "datatype": getattr(self, "datatype", None),
            "internal_model": internal_model,
        }
        return state

    def __setstate__(self, state):
        self._set_internal_model(state.pop("internal_model"))
        self.__dict__.update(state)

    def get_combined_model(self):
        """
        Return single-GPU model for serialization

        Returns
        -------

        model : Trained single-GPU model or None if the model has not
               yet been trained.
        """

        internal_model = self._check_internal_model(self._get_internal_model())

        if isinstance(self.internal_model, Iterable):
            # This function needs to return a single instance of cuml.Base,
            # even if the class is just a composite.
            raise ValueError(
                "Expected a single instance of cuml.Base "
                "but got %s instead." % type(self.internal_model)
            )

        elif isinstance(self.internal_model, Future):
            internal_model = self.internal_model.result()

        return internal_model

    def _set_internal_model(self, model):
        """
        Assigns model (a Future or list of futures containins a single-GPU
        model) to be an internal model.

        This function standardizes upon the way we set the internal model
        so that it could either be futures, a single future, or a class local
        to the client.

        In order for `get_combined model()` to provide a consistent output,
        self.internal_model is expected to be either a single future
        containing a cuml.Base instance or a local cuml.Base on the client.
        An iterable can be passed into this method when a trained model
        has been replicated across the workers. In this case, only the
        first element of the iterable will be set as the internal_model

        If multiple different parameters have been trained across the cluster,
        such as in RandomForests or some approx. nearest neighbors algorithms,
        they should be combined into a single model and the combined model
        should be passed to `set_internal_model()`

        Parameters
        ----------

        model : distributed.client.Future[cuml.Base], cuml.Base, or None

        """
        self.internal_model = self._check_internal_model(model)

    @staticmethod
    def _check_internal_model(model):
        """
        Performs a brief validation that a model meets the requirements
        to be set as an `internal_model`

        Parameters
        ----------

        model : distributed.client.Future[cuml.Base], cuml.Base, or None

        Returns
        -------

        model : distributed.client.Future[cuml.Base], cuml.Base, or None

        """
        if isinstance(model, Iterable):
            # If model is iterable, just grab the first
            model = first(model)

        if isinstance(model, Future):
            if model.type is None:
                wait_and_raise_from_futures([model])

            if not issubclass(model.type, Base):
                raise ValueError(
                    "Dask Future expected to contain cuml.Base "
                    "but found %s instead." % model.type
                )

        elif model is not None and not isinstance(model, Base):
            raise ValueError(
                "Expected model of type cuml.Base but found %s "
                "instead." % type(model)
            )
        return model

    def _get_internal_model(self):
        """
        Gets the internal model from the instance.

        This function is a convenience for future maintenance and
        should never perform any expensive operations like data
        transfers between the client and the Dask cluster.

        Returns
        -------

        internal_model : dask.client.Future[cuml.Base], cuml.Base or None
        """
        return self.internal_model

    @staticmethod
    def _get_model_attr(model, name):
        if hasattr(model, name):
            return getattr(model, name)
        # skip raising an error for ipython/jupyter related attributes
        elif any([x in name for x in ("_ipython", "_repr")]):
            pass
        else:
            raise AttributeError(
                "Attribute %s does not exist on model %s" % (name, type(model))
            )

    def __getattr__(self, attr):
        """
        Method gives access to the correct format of cuml Array attribute to
        the users and proxies attributes to the underlying trained model.

        If the attribute being requested is not directly on the local object,
        this function will see if the local object contains the attribute
        prefixed with an _. In the case the attribute does not exist on this
        local instance, the request will be proxied to self.internal_model and
        will be fetched either locally or remotely depending on whether
        self.internal_model is a local object instance or a future.
        """
        real_name = "_" + attr

        ret_attr = None

        # First check locally for _ prefixed attr
        if real_name in self.__dict__:
            ret_attr = self.__dict__[real_name]

        # Otherwise, if the actual attribute name exists on the
        # object, just return it.
        elif attr in self.__dict__:
            ret_attr = self.__dict__[attr]

        # If we didn't have an attribute on the local model, we might
        # have it on the distributed model.
        internal_model = self._get_internal_model()
        if ret_attr is None and internal_model is not None:
            if isinstance(internal_model, Base):
                # If model is not distributed, just return the
                # requested attribute
                ret_attr = getattr(internal_model, attr)
            else:
                # Otherwise, fetch the attribute from the distributed
                # model and return it
                ret_attr = dask.delayed(BaseEstimator._get_model_attr)(
                    internal_model, attr
                ).compute()
        else:
            raise AttributeError(
                "Attribute %s not found in %s" % (attr, type(self))
            )

        return ret_attr


class DelayedParallelFunc(object):
    def _run_parallel_func(
        self,
        func,
        X,
        n_dims=1,
        delayed=True,
        output_futures=False,
        output_dtype=None,
        output_collection_type=None,
        **kwargs,
    ):
        """
        Runs a function embarrassingly parallel on a set of workers while
        reusing instances of models and constraining the number of
        tasks that can execute concurrently on each worker.

        Note that this mixin assumes the subclass has been trained and
        includes a `self._get_internal_model()` function containing a subclass
        of `cuml.Base`.

        This is intended to abstract functions like predict, transform, and
        score, which can execute embarrassingly parallel but need addition
        execution constraints which result from the more limited GPU
        resources.

        Parameters
        ----------
        func : dask.delayed function to propagate to the workers to execute
               embarrassingly parallel, shared between tasks on each worker

        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).

        delayed : bool return a lazy (delayed) object?

        output_futures : bool returns the futures pointing the to the results
                         of the parallel function executions on the workers,
                         rather than a dask collection object.

        output_collection_type : None or a string in {'cupy', 'cudf'}
            Choose to return the resulting collection as a CuPy backed
            dask.array or a dask_cudf.DataFrame. If None, will use the same
            collection type as used in the input of fit.
            Unused if output_futures=True.

        Returns
        -------
        y : dask cuDF (n_rows, 1)
        """
        X_d = X.to_delayed()

        if output_collection_type is None:
            output_collection_type = self.datatype

        model_delayed = dask.delayed(
            self._get_internal_model(), pure=True, traverse=False
        )

        func = dask.delayed(func, pure=False, nout=1)
        if isinstance(X, dcDataFrame):
            preds = [func(model_delayed, part, **kwargs) for part in X_d]
            dtype = first(X.dtypes) if output_dtype is None else output_dtype
        elif isinstance(X, dcSeries):
            preds = [func(model_delayed, part, **kwargs) for part in X_d]
            dtype = X.dtype if output_dtype is None else output_dtype
        else:
            preds = [func(model_delayed, part[0]) for part in X_d]
            dtype = X.dtype if output_dtype is None else output_dtype

        # TODO: Put the following conditionals in a
        #  `to_delayed_output()` function
        # TODO: Add eager path back in

        if output_collection_type == "cupy":
            # todo: add parameter for option of not checking directly
            shape = (np.nan,) * n_dims
            preds_arr = [
                dask.array.from_delayed(
                    pred,
                    meta=cp.zeros(1, dtype=dtype),
                    shape=shape,
                    dtype=dtype,
                )
                for pred in preds
            ]

            if output_futures:
                return self.client.compute(preds)
            else:
                output = dask.array.concatenate(
                    preds_arr, axis=0, allow_unknown_chunksizes=True
                )
                return output if delayed else output.persist()

        elif output_collection_type == "cudf":
            if output_futures:
                return self.client.compute(preds)
            else:
                output = dask_cudf.from_delayed(preds)
                return output if delayed else output.persist()
        else:
            raise ValueError(
                "Expected cupy or cudf but found %s" % (output_collection_type)
            )


class DelayedPredictionProbaMixin(DelayedParallelFunc):
    def _predict_proba(self, X, delayed=True, **kwargs):
        return self._run_parallel_func(
            func=_predict_proba_func, X=X, n_dims=2, delayed=delayed, **kwargs
        )


class DelayedPredictionMixin(DelayedParallelFunc):
    def _predict(self, X, delayed=True, **kwargs):
        return self._run_parallel_func(
            func=_predict_func, X=X, n_dims=1, delayed=delayed, **kwargs
        )


class DelayedTransformMixin(DelayedParallelFunc):
    def _transform(self, X, n_dims=1, delayed=True, **kwargs):
        return self._run_parallel_func(
            func=_transform_func, X=X, n_dims=n_dims, delayed=delayed, **kwargs
        )


class DelayedInverseTransformMixin(DelayedParallelFunc):
    def _inverse_transform(self, X, n_dims=1, delayed=True, **kwargs):
        return self._run_parallel_func(
            func=_inverse_transform_func,
            X=X,
            n_dims=n_dims,
            delayed=delayed,
            **kwargs,
        )


class SyncFitMixinLinearModel(object):
    def _fit(self, model_func, data):
        n_cols = data[0].shape[1]

        data = DistributedDataHandler.create(data=data, client=self.client)
        self.datatype = data.datatype

        comms = Comms(comms_p2p=False)
        comms.init(workers=data.workers)

        data.calculate_parts_to_sizes(comms)
        self.ranks = data.ranks

        worker_info = comms.worker_info(comms.worker_addresses)
        parts_to_sizes, _ = parts_to_ranks(
            self.client, worker_info, data.gpu_futures
        )

        lin_models = dict(
            [
                (
                    data.worker_info[worker_data[0]]["rank"],
                    self.client.submit(
                        model_func,
                        comms.sessionId,
                        self.datatype,
                        **self.kwargs,
                        pure=False,
                        workers=[worker_data[0]],
                    ),
                )
                for worker, worker_data in enumerate(
                    data.worker_to_parts.items()
                )
            ]
        )

        fit_func = self._func_fit
        lin_fit = dict(
            [
                (
                    worker_data[0],
                    self.client.submit(
                        fit_func,
                        lin_models[data.worker_info[worker_data[0]]["rank"]],
                        worker_data[1],
                        data.total_rows,
                        n_cols,
                        parts_to_sizes,
                        data.worker_info[worker_data[0]]["rank"],
                        pure=False,
                        workers=[worker_data[0]],
                    ),
                )
                for worker, worker_data in enumerate(
                    data.worker_to_parts.items()
                )
            ]
        )

        wait_and_raise_from_futures(list(lin_fit.values()))

        comms.destroy()
        return lin_models

    @staticmethod
    def _func_fit(f, data, n_rows, n_cols, partsToSizes, rank):
        return f.fit(data, n_rows, n_cols, partsToSizes, rank)


def mnmg_import(func):
    @wraps(func)
    def check_cuml_mnmg(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError:
            raise RuntimeError(
                "cuML has not been built with multiGPU support "
                "enabled. Build with the --multigpu flag to"
                " enable multiGPU support."
            )

    return check_cuml_mnmg


def _predict_func(model, data, **kwargs):
    return model.predict(data, **kwargs)


def _predict_proba_func(model, data, **kwargs):
    return model.predict_proba(data, **kwargs)


def _transform_func(model, data, **kwargs):
    return model.transform(data, **kwargs)


def _inverse_transform_func(model, data, **kwargs):
    return model.inverse_transform(data, **kwargs)
