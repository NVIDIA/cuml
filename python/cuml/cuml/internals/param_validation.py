#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Declarative validation of estimator hyperparameters.

Estimators declare the valid values of each hyperparameter they own in a
``_parameter_constraints`` class attribute. Each parameter maps to a list of
constraints, a value is valid if any of them accepts it::

    class Ridge(Base):
        _parameter_constraints = {
            "alpha": [Float(ge=0), ArrayLike],
            "fit_intercept": [Bool],
            "solver": [Options("auto", "eig", "svd", "lsmr")],
            "max_iter": [Int(ge=1), None],
        }

Besides constraint objects, the lists may contain ``None`` and classes
(e.g. ``dict``), which become an ``isinstance`` check. ``bool``, ``int`` and
``float`` are rejected because ``isinstance`` gives surprising results for
them (``True`` is an ``int``, ``np.float32(1.0)`` is not a ``float``); use
``Bool``, ``Int()`` and ``Float()`` instead. Constraints declared by parent
classes are inherited. Validation happens at the start of every fit-like
method (see :func:`cuml.internals.mlfunc`) and raises
:class:`InvalidParameterError`.

Checks that relate several parameters to each other, or parameters to the
input data, are not expressed here and stay in the estimator's code.
"""

import functools
import math
import numbers

import cupy as cp
import numpy as np

__all__ = (
    "InvalidParameterError",
    "Constraint",
    "Int",
    "Float",
    "Options",
    "AnyOf",
    "Bool",
    "ArrayLike",
    "Callable",
    "RandomState",
    "as_constraint",
    "get_parameter_constraints",
    "validate_params",
)


class InvalidParameterError(ValueError, TypeError):
    """Raised when an estimator hyperparameter has an invalid value.

    Inherits from both ``ValueError`` and ``TypeError`` so that code catching
    either keeps working.
    """


class Constraint:
    """A set of valid values for a hyperparameter.

    Subclasses implement ``accepts``, ``__str__`` (used in error messages)
    and optionally ``invalid_examples`` (used by the test suite).
    """

    def accepts(self, value) -> bool:
        raise NotImplementedError

    def invalid_examples(self) -> list:
        """Values that this constraint rejects, for use in tests."""
        return []

    def __repr__(self):
        return f"<{type(self).__name__}: {self}>"


def _type_name(cls) -> str:
    """The name of ``cls`` as used in error messages (same as sklearn)."""
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


class _Number(Constraint):
    """A number, optionally bounded on either side."""

    _description: str
    _name: str

    def __init__(self, *, ge=None, gt=None, le=None, lt=None):
        if ge is not None and gt is not None:
            raise ValueError("Only one of `ge` and `gt` may be set")
        if le is not None and lt is not None:
            raise ValueError("Only one of `le` and `lt` may be set")
        self.ge = ge
        self.gt = gt
        self.le = le
        self.lt = lt

    def _is_type(self, value) -> bool:
        raise NotImplementedError

    def accepts(self, value) -> bool:
        return (
            self._is_type(value)
            and (self.ge is None or value >= self.ge)
            and (self.gt is None or value > self.gt)
            and (self.le is None or value <= self.le)
            and (self.lt is None or value < self.lt)
        )

    def _format_bound(self, bound):
        return bound

    def __str__(self):
        # Formatted the same as sklearn's `Interval`
        low = self.gt if self.ge is None else self.ge
        high = self.lt if self.le is None else self.le
        if low is None and high is None:
            return f"an instance of {self._name!r}"
        left = "(" if self.ge is None else "["
        right = ")" if self.le is None else "]"
        low = "-inf" if low is None else self._format_bound(low)
        high = "inf" if high is None else self._format_bound(high)
        return f"{self._description} in the range {left}{low}, {high}{right}"


class Int(_Number):
    """An integer (including numpy integers, excluding bools)."""

    _description = "an int"
    _name = "int"

    def _is_type(self, value) -> bool:
        return isinstance(value, numbers.Integral) and not isinstance(
            value, bool
        )

    def invalid_examples(self) -> list:
        examples = [1.5, True]
        if self.ge is not None:
            examples.append(self.ge - 1)
        if self.gt is not None:
            examples.append(self.gt)
        if self.le is not None:
            examples.append(self.le + 1)
        if self.lt is not None:
            examples.append(self.lt)
        return examples


class Float(_Number):
    """A real number (including ints and numpy floats, excluding bools and NaN)."""

    _description = "a float"
    _name = "float"

    def _format_bound(self, bound):
        return float(bound)

    def _is_type(self, value) -> bool:
        return (
            isinstance(value, numbers.Real)
            and not isinstance(value, (bool, np.bool_))
            and not math.isnan(value)
        )

    def invalid_examples(self) -> list:
        examples = [math.nan, True]
        if self.ge is not None:
            examples.append(math.nextafter(self.ge, -math.inf))
        if self.gt is not None:
            examples.append(self.gt)
        if self.le is not None:
            examples.append(math.nextafter(self.le, math.inf))
        if self.lt is not None:
            examples.append(self.lt)
        return examples


class Options(Constraint):
    """One of a fixed set of values, typically strings."""

    def __init__(self, *values):
        self.values = values
        self._values = frozenset(values)

    def accepts(self, value) -> bool:
        try:
            return value in self._values
        except TypeError:
            # Unhashable values (e.g. arrays) are never an option
            return False

    def __str__(self):
        values = ", ".join(map(repr, self.values))
        return f"a {_type_name(type(self.values[0]))} among {{{values}}}"

    def invalid_examples(self) -> list:
        return [f"not {self.values[0]}"]


class _InstanceOf(Constraint):
    """An instance of the given class."""

    def __init__(self, cls):
        self.cls = cls

    def accepts(self, value) -> bool:
        return isinstance(value, self.cls)

    def __str__(self):
        return f"an instance of {_type_name(self.cls)!r}"


class _Bool(Constraint):
    def accepts(self, value) -> bool:
        return isinstance(value, (bool, np.bool_))

    def __str__(self):
        return (
            f"an instance of {_type_name(bool)!r} or "
            f"an instance of {_type_name(np.bool_)!r}"
        )

    def invalid_examples(self) -> list:
        return [1]


class _None(Constraint):
    def accepts(self, value) -> bool:
        return value is None

    def __str__(self):
        return "None"


class _ArrayLike(Constraint):
    def accepts(self, value) -> bool:
        if isinstance(value, (list, tuple)):
            return True
        return not np.isscalar(value) and (
            hasattr(value, "__array__")
            or hasattr(value, "__array_interface__")
            or hasattr(value, "__cuda_array_interface__")
        )

    def __str__(self):
        return "an array-like"


class _Callable(Constraint):
    def accepts(self, value) -> bool:
        return callable(value)

    def __str__(self):
        return "a callable"


class AnyOf(Constraint):
    """Accepts a value if any of its member constraints accepts it."""

    def __init__(self, *constraints):
        members = []
        for c in map(as_constraint, constraints):
            members.extend(c.constraints if isinstance(c, AnyOf) else [c])
        self.constraints = members

    def accepts(self, value) -> bool:
        return any(c.accepts(value) for c in self.constraints)

    def __str__(self):
        descriptions = [str(c) for c in self.constraints]
        if len(descriptions) == 1:
            return descriptions[0]
        return f"{', '.join(descriptions[:-1])} or {descriptions[-1]}"

    def invalid_examples(self) -> list:
        # An example rejected by one member may be accepted by another
        return [
            v
            for c in self.constraints
            for v in c.invalid_examples()
            if not self.accepts(v)
        ]


# `isinstance` checks against these don't match what users expect
_BUILTIN_REPLACEMENTS = {bool: "Bool", int: "Int()", float: "Float()"}


def as_constraint(spec) -> Constraint:
    """Convert a declared constraint specification to a ``Constraint``."""
    if isinstance(spec, Constraint):
        return spec
    elif spec is None:
        return _None()
    elif isinstance(spec, type):
        if (replacement := _BUILTIN_REPLACEMENTS.get(spec)) is not None:
            raise TypeError(
                f"Unsupported parameter constraint {spec!r}, use "
                f"{replacement} instead"
            )
        return _InstanceOf(spec)
    raise TypeError(f"Unsupported parameter constraint {spec!r}")


Bool = _Bool()
ArrayLike = _ArrayLike()
Callable = _Callable()

# Matches the values accepted by `cuml.internals.validation.check_random_seed`
RandomState = AnyOf(
    Int(ge=0, le=2**32 - 1),
    np.random.RandomState,
    cp.random.RandomState,
    None,
)


@functools.cache
def get_parameter_constraints(cls) -> dict[str, Constraint]:
    """The constraints for ``cls``, merged with those of its parent classes."""
    merged = {}
    for klass in reversed(cls.__mro__):
        merged.update(vars(klass).get("_parameter_constraints", {}))
    out = {}
    for name, specs in merged.items():
        if not isinstance(specs, list):
            raise TypeError(
                f"The constraints for {cls.__name__}.{name} must be a list, "
                f"got {specs!r}"
            )
        out[name] = AnyOf(*specs)
    return out


def validate_params(estimator):
    """Validate the hyperparameters of ``estimator`` against its constraints.

    Parameters without a declared constraint are not checked.

    Raises
    ------
    InvalidParameterError
        If a hyperparameter value is not accepted by its constraint.
    """
    constraints = get_parameter_constraints(type(estimator))
    for name, value in estimator.get_params(deep=False).items():
        constraint = constraints.get(name)
        if constraint is not None and not constraint.accepts(value):
            raise InvalidParameterError(
                f"The {name!r} parameter of {type(estimator).__name__} must "
                f"be {constraint}. Got {value!r} instead."
            )
