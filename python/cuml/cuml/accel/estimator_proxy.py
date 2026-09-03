#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import abc
import functools
from typing import Any

import cupy as cp
import numpy as np
import sklearn
from cupyx.scipy.sparse import issparse as is_cp_sparse
from packaging.version import Version
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    OneToOneFeatureMixin,
)
from sklearn.utils._set_output import (
    _get_output_config,
    _wrap_data_with_container,
)

from cuml.accel import profilers
from cuml.accel.core import logger
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU, is_fitted
from cuml.internals.outputs import mlfunc, using_output_type
from cuml.internals.validation import check_inputs

__all__ = ("ProxyBase", "ArrayAPIProxyBase", "is_proxy")


SKLEARN_18 = Version(sklearn.__version__) >= Version("1.8.0.dev0")
SKLEARN_110 = Version(sklearn.__version__) >= Version("1.10.0.dev0")


class classproperty:
    """A property, but on the class instead of the instance."""

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def is_proxy(instance_or_class) -> bool:
    """Check if an instance or class is a proxy object created by the accelerator."""
    if isinstance(instance_or_class, type):
        cls = instance_or_class
    else:
        cls = type(instance_or_class)
    return isinstance(cls, ProxyBaseMeta) and hasattr(cls, "_cpu_class")


def ensure_host(x):
    """Convert any cupy/cupyx.scipy.sparse inputs to their host equivalents"""
    return x.get() if (isinstance(x, cp.ndarray) or is_cp_sparse(x)) else x


class _ReconstructProxy:
    """A function for reconstructing serialized estimators.

    Defined this way so that it pickles by value, allowing unpickling in
    environments where cuml isn't installed."""

    @functools.cached_property
    def _reconstruct(self):
        # The reconstruct function is defined in a closure so that cloudpickle
        # will always serialize it by value rather than by reference. This allows
        # the saved model to be loaded in an environment without cuml installed,
        # where it will load as the CPU model directly.
        def reconstruct(cls_path, cpu_model, load_on_gpu):
            """Reconstruct a serialized estimator.

            Returns a proxy estimator if `cuml.accel` is installed, falling back
            to a CPU model otherwise.
            """
            import importlib

            path, _, name = cls_path.rpartition(".")
            module = importlib.import_module(path)
            try:
                # `cuml` is in the exclude list, so this _reconstruct method is as well.
                # To work around this, we directly access the accelerator override mapping
                # (if available), falling back to CPU if not.
                cls = module._accel_overrides[name]
            except (KeyError, AttributeError):
                # Either:
                # - cuml.accel is not installed
                # - The cuml version doesn't support this estimator type
                # Return the CPU estimator directly
                return cpu_model
            # `cuml.accel` is installed, reconstruct a proxy estimator
            return cls._reconstruct_from_cpu(cpu_model, load_on_gpu)

        return reconstruct

    def __reduce__(self):
        import pickle

        # Prefer cloudpickle bundled with joblib (found in joblib < 1.6.0),
        # falling back to cloudpickle otherwise (required for joblib >= 1.6.0).
        # Since joblib is a required dependency of sklearn (and sklearn is a
        # required dep of cuml) one of these should always be available.
        try:
            import joblib.externals.cloudpickle as cloudpickle
        except ImportError:
            import cloudpickle

        return (pickle.loads, (cloudpickle.dumps(self._reconstruct),))

    def __call__(self, cls_path, cpu, load_on_gpu):
        return self._reconstruct(cls_path, cpu, load_on_gpu)


_reconstruct_proxy = _ReconstructProxy()


class ProxyBaseMeta(abc.ABCMeta):
    """A metaclass for `ProxyBase` types.

    Most of the magic of `ProxyBase` lives in `ProxyBase.__init_subclass__`.
    However, to support subclassing proxy estimators (which may make use of
    sklearn internals, and thus cannot be actual proxies), we need a way to
    dynamically modify the bases of a class. We also want these subclasses to
    identify as subclasses (and instances) of the proxy class, even if the
    proxy class isn't a true base.

    Unfortunately, metaclasses don't compose as well - the metaclass of a new
    class must be a (non-strict) subclass of the metaclass of all base classes.
    As such, if any subclasses introduce a new metaclass to do other magic, the
    magic of `ProxyBaseMeta` will cause issues. To work around this, we subclass
    `ProxyBaseMeta` from the most common metaclass in use (`abc.ABCMeta`) so
    these can at least be mixed.
    """

    def __new__(cls, name, bases, ns, **kwargs):
        # If any base classes are ProxyBaseMeta instances _with_ a cpu class
        # defined, replace them with their CPU class.
        bases = tuple(
            getattr(base, "_cpu_class", base)
            if isinstance(base, ProxyBaseMeta)
            else base
            for base in bases
        )
        return super().__new__(cls, name, bases, ns, **kwargs)

    def __subclasscheck__(self, subclass):
        """Check if a class is a subclass"""
        # Check if it's a true subclass
        if super().__subclasscheck__(subclass):
            return True
        # Check if its a subclass of _cpu_class (if available)
        if (cpu_class := getattr(self, "_cpu_class", None)) is not None:
            return cpu_class.__subclasscheck__(subclass)
        # Not a subclass
        return False

    def __instancecheck__(self, instance):
        """Check if an object is an instance."""
        # Check if it's a true instance
        if super().__instancecheck__(instance):
            return True
        # Check if its an instance of _cpu_class (if available)
        if (cpu_class := getattr(self, "_cpu_class", None)) is not None:
            return cpu_class.__instancecheck__(instance)
        # Not an instance
        return False


class ProxyBase(BaseEstimator, metaclass=ProxyBaseMeta):
    """A base class for defining new Proxy estimators.

    Subclasses should define ``_gpu_class``, which must be a subclass of
    ``InteropMixin``.

    Attributes and hyperparameters are always proxied through the CPU estimator.

    Methods _may_ be run on the GPU estimator if:

    - The GPU estimator supports all the hyperparameters
    - The GPU estimator has a method with the same name

    Additionally, GPU-specific behavior for a method may be overridden by defining a
    method on this class with the name ``_gpu_{method}`` (and the same signature).
    If defined, this will be called instead of ``self._gpu.{method}`` when dispatching
    to the GPU with the rules above. If this method raises a ``UnsupportedOnGPU`` error
    then the proxy will fallback to CPU.

    See the definitions in ``cuml.accel._overrides.linear_model`` for examples.
    """

    # A set of attribute names that aren't supported by `cuml.accel`.
    # Attributes in this set will raise a nicer error message than the default
    # AttributeError.
    _not_implemented_attributes = frozenset()

    # A set of additional attribute names to proxy through that don't match the
    # `*_` naming convention. Typically this is private attributes that
    # consumers might still be using.
    _other_attributes = frozenset()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The CPU estimator, always defined. This is the source of truth for any
        # hyperparameters. It also may have fit attributes on it - these are lazily
        # synced from the GPU estimator when accessed.
        self._cpu = self._cpu_class(*args, **kwargs)
        # The GPU estimator. This is only non-None if an estimator was successfully fit
        # on the GPU.
        self._gpu = None
        # Whether fit attributes on the CPU estimator are currently in sync with those
        # on the GPU estimator.
        self._synced = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "_gpu_class"):
            # _gpu_class not defined, assume intermediate class and return
            return

        # Store `_cpu_class` from `_gpu_class` for parity and ease-of-reference
        cls._cpu_class = cls._gpu_class._get_cpu_class()

        # Wrap __init__ to ensure signature compatibility.
        orig_init = cls.__init__
        if cls._cpu_class.__init__ is object.__init__:
            # XXX: Python < 3.13 `inspect.signature` has a bug where a wrapped
            # version of `object.__init__` will display `*args, **kwargs`,
            # while the original `object.__init__` won't. Here we special case
            # estimators with not parameters to work around this. This can be
            # removed once we drop support for Python < 3.13.
            @functools.wraps(cls._cpu_class.__init__)
            def __init__(self):
                orig_init(self)

            del __init__.__wrapped__
        else:

            @functools.wraps(cls._cpu_class.__init__)
            def __init__(self, *args, **kwargs):
                orig_init(self, *args, **kwargs)

        cls.__init__ = __init__

        # Forward some metadata to make the proxy class look like the CPU class
        cls.__qualname__ = cls._cpu_class.__qualname__
        cls.__doc__ = cls._cpu_class.__doc__
        # We use the `_cpu_class_path` (which matches the patched path) instead of
        # the original full module path so that the class (not an instance) can
        # be pickled properly.
        cls.__module__ = cls._gpu_class._cpu_class_path.rsplit(".", 1)[0]

        # Drop `__firstlineno__`, since we can't give a good option for that.
        # We don't want to expose the original value, since we've changed the
        # defining `__module__`. And we don't want to use our `__firstlineno__`
        # since it would map to the definitions in the overrides (which don't
        # have the same `__module__` either). We use a `try-except` here since
        # the __firstlineno__ may be stored in `cls.__dict__` (the one we want
        # to drop) or `type(cls).__dict__` (which will error), depending on
        # cPython version.
        try:
            del cls.__firstlineno__
        except AttributeError:
            pass

        # Forward a few optional class attributes if defined. We do a type
        # check on them for sanity and to avoid forwarding properties.
        for name, typ in [
            ("_estimator_type", str),
            ("_parameter_constraints", dict),
        ]:
            if isinstance(val := getattr(cls._cpu_class, name, None), typ):
                setattr(cls, name, val)

        # All transformer _classes_ have `set_output` defined and gated with
        # `@available_if`. If `get_feature_names_out` isn't defined, then
        # `set_output` won't be available on an _instance_. We exclude
        # `set_output` in that case.
        exclude = set()
        if not hasattr(cls._cpu_class, "get_feature_names_out"):
            exclude.add("set_output")

        # Add proxy method definitions for all public methods on CPU class
        # that aren't already defined on the proxy class
        methods = [
            name
            for name in dir(cls._cpu_class)
            if not name.startswith("_")
            and callable(getattr(cls._cpu_class, name))
            and name not in exclude
        ]

        def _make_method(name):
            cpu_method = getattr(cls._cpu_class, name)

            @functools.wraps(cpu_method)
            def method(self, *args, **kwargs):
                return self._call_method(name, *args, **kwargs)

            return method

        for name in methods:
            if not hasattr(cls, name):
                setattr(cls, name, _make_method(name))

    def _sync_params_to_gpu(self) -> None:
        """Sync hyperparameters to GPU estimator from CPU estimator.

        If the hyperparameters are unsupported, will cause proxy
        to fallback to CPU until refit."""
        if self._gpu is not None:
            try:
                params = self._gpu_class._params_from_cpu(self._cpu)
                self._gpu.set_params(**params)
                logger.debug(
                    f"`{self._cpu_class.__name__}` parameters synced to GPU"
                )
            except UnsupportedOnGPU as exc:
                reason = str(exc) or "Hyperparameters not supported"
                logger.info(
                    f"`{self._cpu_class.__name__}` parameters failed to sync to GPU, "
                    f"falling back to CPU: {reason}"
                )
                self._sync_attrs_to_cpu()
                self._gpu = None

    def _sync_attrs_to_cpu(self) -> None:
        """Sync fit attributes to CPU estimator from GPU estimator.

        This is a no-op if fit attributes are already in sync.
        """
        if self._gpu is not None and not self._synced:
            self._gpu._sync_attrs_to_cpu(self._cpu)
            self._synced = True
            logger.debug(
                f"`{self._cpu_class.__name__}` fitted attributes synced to CPU"
            )

    def _reset_cpu(self) -> None:
        """Reset `_cpu`, dropping all fitted attributes"""
        # XXX: sklearn.clone doesn't copy `_parent_callback_ctx` (but does copy
        # other private state like `_skl_callbacks`, `_sklearn_output_config`,
        # ...). Since we store the context on the CPU estimator, we want this
        # to persist even when resetting the CPU implementation.
        cpu = sklearn.clone(self._cpu)
        if (
            ctx := getattr(self._cpu, "_parent_callback_ctx", None)
        ) is not None:
            cpu._parent_callback_ctx = ctx
        self._cpu = cpu
        self._synced = False

    @classmethod
    def _reconstruct_from_cpu(cls, cpu, load_on_gpu=True):
        """Reconstruct a proxy estimator from its CPU counterpart.

        Primarily used when unpickling serialized proxy estimators."""
        assert type(cpu) is cls._cpu_class
        self = cls.__new__(cls)
        self._cpu = cpu
        self._synced = False
        if load_on_gpu and is_fitted(self._cpu):
            # This estimator is fit and should be loaded on GPU. Try to convert
            # model back to GPU.
            try:
                self._gpu = self._gpu_class.from_sklearn(self._cpu)
            except UnsupportedOnGPU:
                self._gpu = None
            else:
                # Supported on GPU, clear fit attributes from CPU to release host memory
                self._reset_cpu()
        else:
            # Estimator is unfit or should remain on CPU
            self._gpu = None
        return self

    def _call_gpu_method(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the wrapped GPU estimator."""
        from cuml.common.sparse import is_sparse

        if (
            args
            and is_sparse(args[0])
            and "sparse" not in self._gpu.__sklearn_tags__().X_types_gpu
        ):
            raise UnsupportedOnGPU("Sparse inputs are not supported")

        if getattr(self._cpu, "_skl_callbacks", ()) and method in (
            "fit",
            "fit_transform",
            "fit_predict",
            "partial_fit",
        ):
            raise UnsupportedOnGPU(
                "Estimators with callbacks are not supported"
            )

        # Determine the function to call. Check for an override on the proxy class,
        # falling back to the GPU class method if one exists.
        gpu_func = getattr(self, f"_gpu_{method}", None)
        if gpu_func is None:
            if (gpu_func := getattr(self._gpu, method, None)) is None:
                raise UnsupportedOnGPU("Method is not implemented in cuml")

        # Only transform/fit_transform/inverse_transform with default
        # set_output config may return device arrays (to support optimized
        # pipeline data transfers). All other methods must return on host.
        may_return_on_device = (
            method in ("transform", "fit_transform", "inverse_transform")
            and _get_output_config("transform", self)["dense"] == "default"
        )
        if may_return_on_device:
            out = gpu_func(*args, **kwargs)
        else:
            with using_output_type("numpy"):
                out = gpu_func(*args, **kwargs)

        if method in ("transform", "fit_transform"):
            # Properly wrap output of transform following `set_output` config.
            out = _wrap_data_with_container("transform", out, args[0], self)

        return self if out is self._gpu else out

    def _call_method(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the proxied estimators."""

        if method.startswith("set_") and method.endswith("_request"):
            # This is a metadata request setter (like `set_{method}_request`),
            # always dispatch directly to CPU.
            getattr(self._cpu, method)(*args, **kwargs)
            return self

        qualname = f"{self._cpu_class.__name__}.{method}"

        is_fit = method in (
            "fit",
            "fit_transform",
            "fit_predict",
            "partial_fit",
        )

        reason = None
        if is_fit:
            # Call CPU param validation to validate hyperparameters.
            # This ensures we match errors for invalid hyperparameters during fitting.
            self._validate_params()

            if method == "partial_fit" and is_fitted(self):
                # Partial fit on already fit models should reuse existing state.
                if self._gpu is not None:
                    # GPU partial_fit will invalidate CPU state, reset on CPU
                    self._reset_cpu()
            else:
                # Reinitialize GPU model for the current hyperparameters
                try:
                    self._gpu = self._gpu_class(
                        **self._gpu_class._params_from_cpu(self._cpu)
                    )
                except UnsupportedOnGPU as exc:
                    # Unsupported, fallback to CPU
                    reason = str(exc) or "Hyperparameters not supported"
                    self._gpu = None
                else:
                    # New estimator successfully initialized on GPU, reset on CPU
                    self._reset_cpu()

        if self._gpu is not None:
            # The hyperparameters are supported, try calling the method
            try:
                with profilers.track_gpu_call(qualname):
                    out = self._call_gpu_method(method, *args, **kwargs)
                logger.info(f"`{qualname}` ran on GPU")
                return out
            except UnsupportedOnGPU as exc:
                reason = str(exc) or "Method parameters not supported"
                # Unsupported. If it's a `fit` we need to clear
                # the GPU state before falling back to CPU.
                if is_fit:
                    if method == "partial_fit":
                        # For `partial_fit` we want to sync any existing state
                        # before dropping the GPU model
                        self._sync_attrs_to_cpu()
                    self._gpu = None

        if reason is not None:
            logger.info(
                f"`{self._cpu_class.__name__}.{method}` falling back to CPU: {reason}"
            )

        # Failed to run on GPU, fallback to CPU
        self._sync_attrs_to_cpu()
        # Ensure the arguments are on host for the CPU fallback. This is _usually_
        # already True, but in certain cases (a pipeline with optimized data transfer)
        # we may need to migrate. In those cases the inputs will only ever be
        # cupy/cupyx.scipy.sparse objects, so that's all we need to handle here.
        args = [ensure_host(a) for a in args]
        kwargs = {k: ensure_host(v) for k, v in kwargs.items()}
        with profilers.track_cpu_call(
            qualname, reason=reason or "Estimator not fit on GPU"
        ):
            out = getattr(self._cpu, method)(*args, **kwargs)
        logger.info(f"`{qualname}` ran on CPU")
        return self if out is self._cpu else out

    ############################################################
    # set_output handling                                      #
    ############################################################

    def _gpu_set_output(self, *, transform=None):
        # `set_output` can always call the CPU model (where the output config state
        # is stored). It's defined as a `_gpu_*` method so it only shows up on the
        # proxy for models that define `set_output`, and can avoid unnecessary calls
        # to sync fit attributes to CPU
        self._cpu.set_output(transform=transform)
        return self._gpu

    def _gpu_get_feature_names_out(self, input_features=None):
        # In the common case `get_feature_names_out` doesn't require fitted attributes
        # on the CPU. Here we detect and special case common mixins, falling back to
        # CPU when necessary. This helps avoid unnecessary device -> host transfers.
        cpu_method = self._cpu_class.get_feature_names_out
        if cpu_method is ClassNamePrefixFeaturesOutMixin.get_feature_names_out:
            # Can run cpu method directly on GPU instance, it only references `_n_features_out`
            return cpu_method(self._gpu, input_features=input_features)
        if cpu_method is OneToOneFeatureMixin.get_feature_names_out:
            # Uses n_features_in_ (and optionally feature_names_in_) on the estimator.
            # cuML models set n_features_in_ on fit; feature_names_in_ is optional.
            return cpu_method(self._gpu, input_features=input_features)

        # Fallback to CPU
        raise UnsupportedOnGPU

    @property
    def _sklearn_output_config(self):
        # Used by sklearn to handle wrapping output type, just proxy through to the CPU model
        return self._cpu._sklearn_output_config

    @property
    def _sklearn_auto_wrap_output_keys(self):
        # Used by sklearn to handle wrapping output type, just proxy through to the CPU model
        return self._cpu._sklearn_auto_wrap_output_keys

    ############################################################
    # set_callbacks handling                                   #
    ############################################################

    def _gpu_set_callbacks(self, *callbacks):
        self._cpu.set_callbacks(*callbacks)
        return self._gpu

    if SKLEARN_110:
        # Required for sklearn 1.10 callback support
        def _set_callbacks(self, callbacks):
            self._cpu._set_callbacks(callbacks)
            return self

    ############################################################
    # Standard magic methods                                   #
    ############################################################

    def __dir__(self) -> list[str]:
        # Ensure attributes are synced so they show up
        self._sync_attrs_to_cpu()
        return dir(self._cpu)

    def __reduce__(self):
        # We only use the CPU estimator for pickling, ensure its fully synced
        self._sync_attrs_to_cpu()
        return (
            _reconstruct_proxy,
            (
                self._gpu_class._cpu_class_path,
                self._cpu,
                self._gpu is not None,
            ),
        )

    def __getattr__(self, name: str) -> Any:
        is_private = name.startswith("_")

        if (
            name.endswith("_")
            and not is_private
            or name in self._other_attributes
        ):
            # Fit attributes require syncing
            self._sync_attrs_to_cpu()

            try:
                return getattr(self._cpu, name)
            except AttributeError:
                if name in self._not_implemented_attributes and is_fitted(
                    self._cpu
                ):
                    raise AttributeError(
                        f"The `{type(self).__name__}.{name}` attribute is not yet "
                        "implemented in `cuml.accel`.\n\n"
                        "If this attribute is important for your use case, please open "
                        "an issue: https://github.com/NVIDIA/cuml/issues."
                    ) from None
                raise
        elif name in ("_parent_callback_ctx", "_skl_callbacks"):
            # sklearn.callback state always goes through CPU estimator
            return getattr(self._cpu, name)
        elif is_private:
            # Don't proxy magic methods or private attributes by default
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )

        return getattr(self._cpu, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_cpu", "_gpu", "_synced"):
            # Internal state
            object.__setattr__(self, name, value)
        elif name == "_parent_callback_ctx":
            # sklearn.callback state always goes through CPU estimator
            self._cpu._parent_callback_ctx = value
        elif name in self._cpu._get_param_names():
            # Hyperparameter, set on CPU
            setattr(self._cpu, name, value)
            self._sync_params_to_gpu()
        else:
            # Mutating non-hyperparameter (probably a fit attribute). This
            # is weird to do, and should never be done during normal sklearn
            # usage. This does happen sometimes in sklearn tests though.
            # The only sane thing to do is to fallback to CPU and forward
            # the mutation through.
            self._sync_attrs_to_cpu()
            self._gpu = None
            setattr(self._cpu, name, value)

    def __delattr__(self, name: str) -> None:
        if name in ("_cpu", "_gpu", "_synced"):
            # Internal state. We never call this, just here for parity.
            object.__delattr__(self, name)
        elif name == "_parent_callback_ctx":
            # sklearn.callback state always goes through CPU estimator
            del self._cpu._parent_callback_ctx
        else:
            # No normal workflow deletes attributes (hyperparameters or otherwise)
            # from a sklearn estimator. The sklearn tests do sometimes though.
            # The only sane thing to do is to fallback to CPU and forward
            # the mutation through.
            self._sync_attrs_to_cpu()
            self._gpu = None
            delattr(self._cpu, name)

    ############################################################
    # Public sklearn methods                                   #
    ############################################################

    @functools.wraps(BaseEstimator.get_params)
    def get_params(self, deep=True):
        return self._cpu.get_params(deep=deep)

    @functools.wraps(BaseEstimator.set_params)
    def set_params(self, **kwargs):
        self._cpu.set_params(**kwargs)
        self._sync_params_to_gpu()
        return self

    def __sklearn_tags__(self):
        return self._cpu.__sklearn_tags__()

    def __sklearn_is_fitted__(self):
        model = self._cpu if self._gpu is None else self._gpu
        return is_fitted(model)

    def __sklearn_clone__(self):
        cls = type(self)
        out = cls.__new__(cls)
        # Clone only copies hyperparameters.
        out._cpu = sklearn.clone(self._cpu)
        out._gpu = None
        out._synced = False
        return out

    ############################################################
    # Metadata Routing Methods                                 #
    ############################################################

    @functools.wraps(BaseEstimator.get_metadata_routing)
    def get_metadata_routing(self):
        return self._cpu.get_metadata_routing()

    @functools.wraps(BaseEstimator._get_metadata_request)
    def _get_metadata_request(self):
        return self._cpu._get_metadata_request()

    if SKLEARN_18:

        @classmethod
        @functools.wraps(
            BaseEstimator._get_class_level_metadata_request_values
        )
        def _get_class_level_metadata_request_values(cls, method):
            return cls._cpu_class._get_class_level_metadata_request_values(
                method
            )
    else:

        @classmethod
        @functools.wraps(BaseEstimator._get_default_requests)
        def _get_default_requests(cls):
            return cls._cpu_class._get_default_requests()

    @property
    def _metadata_request(self):
        return self._cpu._metadata_request

    ############################################################
    # Methods on BaseEstimator used internally by sklearn      #
    ############################################################

    @property
    def _estimator_type(self):
        return self._cpu._estimator_type

    @classmethod
    def _get_param_names(cls):
        return cls._cpu_class._get_param_names()

    def _validate_params(self):
        if hasattr(self._cpu, "_validate_params") and hasattr(
            self._cpu, "_parameter_constraints"
        ):
            self._cpu._validate_params()

    def _html_repr(self):
        self._sync_attrs_to_cpu()
        return self._cpu._html_repr()


class _ArrayAPIWrapper(InteropMixin, Base):
    """Wraps an array-api enabled sklearn estimator as a cuml estimator.

    This is a **bare-bones implementation**, implementing just enough features
    required to then re-wrap with a `cuml.accel.estimator_proxy.ProxyBase`.
    This lets us run certain sklearn models that support the array-api through
    the normal cuml-accel machinery without having to define custom
    cuml classes for them.
    """

    def __init__(self, *args, output_type=None, verbose=False, **kwargs):
        super().__init__(output_type=output_type, verbose=verbose)
        self._internal_model = self._internal_class(*args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Store `_internal_class` for ease-of-reference
        cls._internal_class = cls._get_cpu_class()

        # Wrap __init__ to ensure signature compatibility.
        orig_init = cls.__init__

        @functools.wraps(cls._internal_class.__init__)
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)

        cls.__init__ = __init__

        def _make_method(name):
            sk_method = getattr(cls._internal_class, name)

            @functools.wraps(sk_method)
            def method(self, *args, **kwargs):
                return self._call_method(name, *args, **kwargs)

            return method

        # Iterate through common method names, and proxy those that we know
        for name in [
            "fit",
            "fit_transform",
            "fit_predict",
            "partial_fit",
            "transform",
            "inverse_transform",
            "predict",
            "predict_log_proba",
            "predict_proba",
            "decision_function",
            "score",
        ]:
            if hasattr(cls._internal_class, name):
                setattr(cls, name, _make_method(name))

    @mlfunc(array_arg="X")
    def _call_method(self, name, X, *args, **kwargs):
        """Call method `name` on the wrapped sklearn estimator."""
        if name in ("fit", "fit_transform", "fit_predict"):
            reset = True
        elif name == "partial_fit":
            reset = not hasattr(self._internal_model, "n_samples_seen_")
        else:
            reset = False

        # Convert X to cupy, with minimal other validation. We enumerate the supported
        # input dtypes here, so `check_inputs` will coerce numeric object inputs
        # to float64 (mirroring sklearn's behavior), while letting other dtypes
        # through unchanged.
        # Note that `feature_names_in_` is set on the _ArrayAPIWrapper model
        # and not the internal sklearn model. The internal model only ever sees
        # cupy array inputs, all coercion to/from pandas happens in the wrapper.
        X = check_inputs(
            self,
            X,
            ensure_all_finite=False,
            ensure_min_samples=0,
            dtype=(
                "float64",
                "float32",
                "float16",
                "int64",
                "int32",
                "int16",
                "int8",
                "uint64",
                "uint32",
                "uint16",
                "uint8",
                "bool",
            ),
            reset=reset,
        )

        # Run the method with array-api enabled, and global transform output
        # set to default. This overrides any user-set global default, so
        # coercion to other output types happens in the wrapper and not the
        # internal model.
        with sklearn.config_context(
            array_api_dispatch=True, transform_output="default"
        ):
            method = getattr(self._internal_model, name)
            out = method(X, *args, **kwargs)
        return self if out is self._internal_model else out

    @classmethod
    def _get_param_names(cls):
        return cls._internal_class._get_param_names()

    @classmethod
    def _params_from_cpu(cls, model):
        if not SKLEARN_18:
            raise UnsupportedOnGPU(
                "scikit-learn >= 1.8 is required to run on GPU"
            )

        if cls._params_from_cpu_override is not None:
            return cls._params_from_cpu_override(model)
        return model.get_params(deep=False)

    def _params_to_cpu(self):
        return self.get_params(deep=False)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.X_types_gpu = ["2darray"]
        return tags

    def _sync_attrs_from_cpu(self, model) -> None:
        if not is_fitted(model):
            # Not fitted, nothing to do
            return

        attrs = self._attrs_from_cpu(model)

        for name, value in attrs.items():
            if name == "n_features_in_":
                # n_features_in_ is set on both wrapper and internal model.
                setattr(self._internal_model, name, value)
                setattr(self, name, value)
            elif name == "feature_names_in_":
                # feature_names_in_ is only set on the wrapper.
                # If it was set on the internal model, the user would see warnings
                # on transform, since the internal model only ever gets cupy inputs.
                setattr(self, name, value)
            else:
                # other attributes are only set on internal model.
                setattr(self._internal_model, name, value)

    def _attrs_from_cpu(self, model):
        attrs = super()._attrs_from_cpu(model)
        exclude = {
            "feature_names_in_",
            "n_features_in_",
            *self._get_param_names(),
        }
        for name, value in vars(model).items():
            if name not in exclude:
                if isinstance(value, np.ndarray):
                    value = cp.asarray(value)
                attrs[name] = value
        return attrs

    def _attrs_to_cpu(self, model):
        attrs = super()._attrs_to_cpu(model)
        exclude = set(self._get_param_names())
        for name, value in vars(self._internal_model).items():
            if name not in exclude:
                if isinstance(value, cp.ndarray):
                    value = cp.asnumpy(value)
                attrs[name] = value
        return attrs

    @functools.wraps(Base.set_params)
    def set_params(self, **kwargs):
        self._internal_model.set_params(**kwargs)
        return self

    @functools.wraps(Base.get_params)
    def get_params(self, deep=True):
        return self._internal_model.get_params(deep=deep)

    def __getattr__(self, name):
        # Don't proxy through fitted or private attributes
        if name.endswith("_") or name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(self._internal_model, name)


class ArrayAPIProxyBase(ProxyBase):
    """A ProxyBase subclass for proxying array-api-enabled sklearn models.

    Subclasses should define ``_cpu_class_path`` as the public import path of
    the sklearn class. They also may optionally define `_params_from_cpu` to
    handle filtering any unsupported hyperparameters.
    """

    def __init_subclass__(cls, **kwargs):
        # If _cpu_class_path not defined, skip generation of accelerated class
        if hasattr(cls, "_cpu_class_path"):
            # Programmatically create a new private cuml.Base class that wraps the
            # sklearn array-api-enabled model in a cuml consistent API.
            cls._gpu_class = type(
                cls.__name__,
                (_ArrayAPIWrapper,),
                {
                    "_cpu_class_path": cls._cpu_class_path,
                    "_params_from_cpu_override": getattr(
                        cls, "_params_from_cpu", None
                    ),
                },
            )
        super().__init_subclass__(**kwargs)
