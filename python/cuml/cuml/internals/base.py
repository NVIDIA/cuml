#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import inspect
import os
import re
import threading

import pylibraft.common.handle

import cuml
import cuml.common
import cuml.internals
import cuml.internals.logger as logger
import cuml.internals.nvtx as nvtx
from cuml.internals.mixins import TagsMixin, _ensure_transformer_tags
from cuml.internals.outputs import infer_output_type

_THREAD_STATE = threading.local()


def get_handle(*, n_streams=0, device_ids=None):
    """Get a `pylibraft.common.Handle`.

    Parameters
    ----------
    n_streams : int, default=0
        The number of streams to use for a backing stream pool. If non-zero
        a temporary `Handle` with a pool that size will be created. Otherwise
        the default threadlocal `Handle` will be used.
    device_ids : list[int], "all", or None, default=None
        If non-None, will return a `pylibraft.common.DeviceResourcesSNMG`,
        enabling multi-device execution. May be a list of device IDs, or
        ``"all"`` to use all available devices.
    """
    if n_streams == 0 and device_ids is None:
        if not hasattr(_THREAD_STATE, "handle"):
            _THREAD_STATE.handle = pylibraft.common.handle.Handle()
        return _THREAD_STATE.handle
    elif device_ids is not None:
        if n_streams != 0:
            # DeviceResourcesSNMG doesn't support `n_streams` at this time
            raise ValueError(
                "Cannot specify both `device_ids` and `n_streams`"
            )
        return pylibraft.common.handle.DeviceResourcesSNMG(
            device_ids=(None if device_ids == "all" else device_ids)
        )
    else:
        return pylibraft.common.handle.Handle(n_streams=n_streams)


class Base(TagsMixin):
    """Base class for cuml estimators.

    Subclasses should:

    - Define ``_get_param_names`` to extend the base implementation with
      any additional parameter names.

    - Decorate their ``fit`` method with
      ``cuml.internals.mlfunc(set_input_type=True)`` to store their fitted
      input type.

    - Decorate methods that return array likes with ``cuml.internals.mlfunc``
      to properly coerce outputs to the proper type. In most cases you'll
      also want to set ``preserve_index=True`` so the index of the input
      dataframe is attached to the output.

    Parameters
    ----------
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    .. code-block:: python

        import cupy as cp
        from cuml.internals import Base, mlfunc

        class MyAlgo(Base):
            def __init__(
                self,
                *,
                param=123,
                verbose=False,
                output_type=None,
            ):
                super().__init__(verbose=verbose, output_type=output_type)
                self.param = param

            @classmethod
            def _get_param_names(cls):
                return [*super()._get_param_names(), "param"]

            @mlfunc(set_input_type=True)
            def fit(self, X, y):
                # Training logic goes here...
                return self

            @mlfunc(preserve_index=True)
            def predict(self, X):
                # Inference logic goes here...
                return cp.ones(len(X), dtype="int32")
    """

    def __init__(
        self,
        *,
        verbose=False,
        output_type=None,
    ):
        self.verbose = verbose
        self.output_type = output_type
        self._input_type = None

        nvtx_benchmark = os.getenv("NVTX_BENCHMARK")
        if nvtx_benchmark and nvtx_benchmark.lower() == "true":
            self.set_nvtx_annotations()

    def __repr__(self, N_CHAR_MAX=700):
        """
        Pretty prints the arguments of a class using Scikit-learn standard :)
        """
        # Only show parameters whose value differs from the constructor
        # default, sorted by name, matching scikit-learn's behavior.
        # `inspect.signature` (unlike `getfullargspec().args`) includes
        # keyword-only parameters, which all cuML estimators now use. Params
        # returned by `get_params` that aren't constructor arguments (e.g.
        # base params injected into sklearn-derived preprocessors) are skipped.
        init_params = inspect.signature(type(self).__init__).parameters
        changed = {}
        for key, value in self.get_params(deep=False).items():
            if key not in init_params:
                continue
            default = init_params[key].default
            if default is inspect.Parameter.empty or repr(value) != repr(
                default
            ):
                changed[key] = value
        body = ", ".join(
            f"{key}={value!r}" for key, value in sorted(changed.items())
        )
        output = f"{type(self).__name__}({body})"

        # Use bruteforce ellipsis when there are a lot of non-blank characters,
        # mirroring `sklearn.base.BaseEstimator.__repr__`.
        n_nonblank = len("".join(output.split()))
        if n_nonblank > N_CHAR_MAX:
            lim = N_CHAR_MAX // 2  # apprx number of chars to keep on both ends
            regex = r"^(\s*\S){%d}" % lim
            left_lim = re.match(regex, output).end()
            right_lim = re.match(regex, output[::-1]).end()

            if "\n" in output[left_lim:-right_lim]:
                regex += r"[^\n]*\n"
                right_lim = re.match(regex, output[::-1]).end()

            ellipsis = "..."
            if left_lim + len(ellipsis) < len(output) - right_lim:
                output = output[:left_lim] + "..." + output[-right_lim:]

        if hasattr(self, "sk_model_"):
            output += " <sk_model_ attribute used>"
        return output

    @property
    def _verbose_level(self):
        """The current `verbose` setting as a `logger.level_enum`"""
        return logger._verbose_to_level(self.verbose)

    @classmethod
    def _get_param_names(cls):
        """
        Returns a list of hyperparameter names owned by this class. It is
        expected that every child class overrides this method and appends its
        extra set of parameters that it in-turn owns. This is to simplify the
        implementation of `get_params` and `set_params` methods.
        """
        return ["verbose", "output_type"]

    def get_params(self, deep=True):
        """
        Returns a dict of all params owned by this class. If the child class
        has appropriately overridden the `_get_param_names` method and does not
        need anything other than what is there in this method, then it doesn't
        have to override this method
        """
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params):
        """
        Accepts a dict of params and updates the corresponding ones owned by
        this class. If the child class has appropriately overridden the
        `_get_param_names` method and does not need anything other than what is,
        there in this method, then it doesn't have to override this method
        """
        if not params:
            return self
        valid_params = self._get_param_names()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for `{type(self).__name__}`"
                )
            setattr(self, key, value)
        return self

    def _set_output_type(self, inp):
        self._input_type = infer_output_type(inp)

    def _get_output_type(self, inp=None):
        """
        Method to be called by predict/transform methods of inheriting classes.
        Returns the appropriate output type depending on the type of the input,
        class output type and global output type.
        """
        # Default to the global type
        output_type = cuml.global_settings.output_type

        # If not set to an explicit value, use the estimator's setting
        if output_type in (None, "input", "mirror"):
            output_type = self.output_type

        # If input, get the type from the input (if available)
        if output_type in (None, "input"):
            if inp is None:
                # No input value provided, use the estimator input type
                output_type = self._input_type
            else:
                # Determine the output from the input
                output_type = infer_output_type(inp)

        return output_type

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # 'preserves_dtype' tag's Scikit definition currently only applies to
        # transformers and whether the transform method conserves the dtype
        # (in that case returns an empty list, otherwise the dtype it
        # casts to).
        # By default, our transform methods convert to self.dtype, but
        # we need to check whether the tag has been defined already.
        if hasattr(self, "transform"):
            transformer_tags = _ensure_transformer_tags(tags)
            if hasattr(self, "dtype"):
                transformer_tags.preserves_dtype = [self.dtype]
        return tags

    def _repr_mimebundle_(self, **kwargs):
        """Prepare representations used by jupyter kernels to display estimator"""
        from sklearn.utils import estimator_html_repr

        output = {"text/plain": repr(self)}
        output["text/html"] = estimator_html_repr(self)
        return output

    def set_nvtx_annotations(self):
        for func_name in [
            "fit",
            "transform",
            "predict",
            "fit_transform",
            "fit_predict",
        ]:
            if hasattr(self, func_name):
                msg = "{class_name}.{func_name} [{addr}]"
                msg = msg.format(
                    class_name=self.__class__.__module__,
                    func_name=func_name,
                    addr=hex(id(self)),
                )
                msg = msg[5:]  # remove cuml.
                func = getattr(self, func_name)
                func = nvtx.annotate(message=msg, domain="cuml_python")(func)
                setattr(self, func_name, func)
