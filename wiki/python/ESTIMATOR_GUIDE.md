# cuML Python Estimators Developer Guide

This guide documents the patterns expected for new or updated `cuml.Base` estimators.

**Note:** Start with the [Quick Start Guide](#quick-start-guide) and [copyable estimator skeleton](#copyable-estimator-skeleton). The later sections explain the estimator contract in more detail for less common cases.

## Table of Contents

- [Recommended Scikit-Learn Documentation](#recommended-scikit-learn-documentation)
- [API Matching Policy](#api-matching-policy)
- [Quick Start Guide](#quick-start-guide)
   - [Copyable Estimator Skeleton](#copyable-estimator-skeleton)
- [Background](#background)
   - [Array I/O and Output Types in cuML](#array-io-and-output-types-in-cuml)
   - [Ingesting Arrays](#ingesting-arrays)
   - [Returning Arrays](#returning-arrays)
- [Estimator Design](#estimator-design)
   - [Initialization](#initialization)
   - [Implementing `_get_param_names()`](#implementing-_get_param_names)
   - [Estimator Tags and cuML Specific Tags](#estimator-tags-and-cuml-specific-tags)
   - [Estimator Array-Like Attributes](#estimator-array-like-attributes)
   - [Estimator Methods](#estimator-methods)
- [Do's and Don'ts](#dos-and-donts)

## Recommended Scikit-Learn Documentation

Read these scikit-learn references first; cuML estimators follow them unless this guide says otherwise:

1. [Scikit-learn's Estimator Docs](https://scikit-learn.org/stable/developers/develop.html)
   1. Pay particular attention to:
      1. [Instantiation](https://scikit-learn.org/stable/developers/develop.html#instantiation)
      2. [Estimated Attributes](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
      3. [`get_params` and `set_params`](https://scikit-learn.org/stable/developers/develop.html#get-params-and-set-params)
      4. [Cloning](https://scikit-learn.org/stable/developers/develop.html#cloning)
      5. [Estimator tags](https://scikit-learn.org/stable/developers/develop.html#estimator-tags)
2. [Scikit-learn's Docstring Guide](https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-documentation)
   1. Follow this for array-like objects, shapes, dtypes, and default values.

## API Matching Policy

cuML often implements GPU-accelerated versions of estimators from CPU libraries, especially scikit-learn. Match the source API when it is useful, but do not copy API surface blindly.

1. **Match the original API where possible and reasonable**
   - Use identical parameter names, types, and default values
   - Keep method signatures and return types consistent
   - Maintain the same behavior and semantics where possible

2. **Justify and document API deviations**
   - Document all API deviations clearly
   - Avoid arbitrary deviations from the original API

     For example, if the original library uses a parameter named `n_neighbor`, we should not arbitrarily change it to `n_neighbors` in our implementation.

   - Explain necessary deviations with in-code comments

3. **Do not add unused parameters just for parity**
   - Omit source-library parameters that cuML cannot use, unless backwards compatibility requires keeping an existing parameter.

4. **Exact API matching is not required**
   - Consumers who need exact API matching should use `cuml.accel`
   - Focus on providing a consistent and intuitive API rather than exact matching
   - Prioritize performance and GPU-specific optimizations over exact API matching
   - Exact API matching is not required, but arbitrary deviations are not permitted.

## Quick Start Guide

At a high level, all cuML Estimators must:

1. Inherit from `cuml.Base`
   ```python
   from cuml.internals.base import Base

   class MyEstimator(Base):
      ...
   ```

2. Follow the [Scikit-learn estimator developer
   guidelines](https://scikit-learn.org/stable/developers/develop.html)

3. Include the `Base.__init__()` arguments available in the new Estimator's
   `__init__()`

   ```python
   from cuml.internals.base import Base

   class MyEstimator(Base):

      def __init__(self, *, extra_arg=True, verbose=False, output_type=None):
         super().__init__(verbose=verbose, output_type=output_type)
         ...
   ```

4. Declare each public array-like attribute the new Estimator will compute as a
   class variable for automatic array type conversion.

   ```python
   from cuml.internals import ReflectedAttr
   from cuml.internals.base import Base

   class MyEstimator(Base):

      labels_ = ReflectedAttr()

      def __init__(self):
         ...
   ```

5. Use the `@mlfunc` decorator on public API methods that return arrays. Use
   `@mlfunc(set_input_type=True)` on fit-like methods in combination with the
   appropriate `cuml.internals.validation` helpers. Any inference method that
   returns an array with `n_samples` rows (aligned with the input `X`) should
   use `@mlfunc(preserve_index=True)` to align the output with the input's
   index (if any). Other methods should use plain `@mlfunc`:

   ```python
   from cuml.internals import mlfunc
   from cuml.internals.base import Base

   class MyEstimator(Base):

      @mlfunc(set_input_type=True)
      def fit(self, X) -> "MyEstimator":
         ...

      @mlfunc(preserve_index=True)
      def predict(self, X):
         ...
   ```

   See [Estimator Methods](#estimator-methods) for detailed guidance on when to
   use `@mlfunc`.

6. Implement `_get_param_names()` including values returned by
   `super()._get_param_names()`

   ```python
      @classmethod
      def _get_param_names(cls):
         return super()._get_param_names() + [
            "eps",
            "min_samples",
         ]
   ```

7. Override estimator tags only when the defaults are wrong. Prefer existing
   [Mixins](../../python/cuml/cuml/internals/mixins.py) for common capabilities
   such as preferred input order, sparse support, string input, or NaN support.
   See [Estimator Tags and cuML-Specific
   Tags](#estimator-tags-and-cuml-specific-tags) for custom tag overrides.

For most estimators, the checklist and skeleton below are enough. The later
sections explain the contract and uncommon cases.

### Copyable Estimator Skeleton

Use this as a starting point for dense estimators that follow the standard cuML
pattern:

```python
from cuml.internals import mlfunc, ReflectedAttr
from cuml.internals.base import Base
from cuml.internals.validation import check_inputs, check_is_fitted


class MyEstimator(Base):
    result_ = ReflectedAttr()

    def __init__(self, *, extra_arg=True, verbose=False, output_type=None):
        super().__init__(verbose=verbose, output_type=output_type)
        self.extra_arg = extra_arg

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "extra_arg"]

    @mlfunc(set_input_type=True)
    def fit(self, X) -> "MyEstimator":
        X = check_inputs(self, X, order="K", reset=True)
        # Replace this placeholder with estimator training.
        self.result_ = X
        return self

    @mlfunc(preserve_index=True)
    def transform(self, X):
        check_is_fitted(self)
        X = check_inputs(self, X, order="K")
        # Return an array-like object directly; @mlfunc handles conversion.
        return X
```

Fit-like methods should use `@mlfunc(set_input_type=True)` in combination with
the appropriate validation helpers. Pass `reset=True` to validation when
fitting. Methods that return scalars may use `@mlfunc(convert_output=False)` to
fully disable the output conversion code path, though a plain `@mlfunc` would
work as well (no conversion happens for scalars anyway).

## Background

### Array I/O and Output Types in cuML

cuML estimators should validate public inputs with `cuml.internals.validation`
helpers such as `check_inputs`, `check_array`, `check_y`, and
`check_sample_weight`. Prefer `check_inputs` for estimator methods that
validate `X` and optional `y`/`sample_weight` values. These helpers accept the
standard cuML array-like inputs, apply estimator feature metadata checks, and
normalize data to standard CuPy/NumPy or sparse array containers. Sparse
estimators should configure the validation helpers for their supported sparse
formats or follow the sparse-specific validation utilities used by neighboring
sparse estimators.

Internally, dense array data should usually be processed as the standard arrays
returned by validation. Low-level code that needs a specific memory location
should request it explicitly through validation arguments such as `mem_type`.

Public output type conversion is handled by `@mlfunc` and `ReflectedAttr`.
Users choose output types in three ways:

1. Set `output_type` on an estimator, for example `MyEstimator(output_type="numpy")`.
2. Set a global override with `cuml.set_global_output_type("numpy")`.
3. Temporarily set a global override with `cuml.using_output_type("numpy")`.

The global setting stored in `cuml.global_settings.output_type` takes
precedence over an estimator's `output_type`. When neither is set, reflected
estimator methods normally mirror the call input type, and descriptor
attributes mirror the fit-time input type.

Accepted output types are:

 - `None`: No global or estimator override. Reflected estimator methods infer
   from their input or fit-time input type.
 - `"input"`: Mirror the relevant input type.
 - `"cupy"`: Return a CuPy array.
 - `"numpy"`: Return a NumPy array.
 - `"cudf"`: Return a cuDF Series or DataFrame.
 - `"pandas"`: Return a pandas Series or DataFrame.

The internal output type `"cuml"` may appear inside reflected calls.
User-facing code should not set it.

### Ingesting Arrays

When the input array type is not known, the correct and safest way to validate
estimator inputs is using `cuml.internals.validation`. For estimator methods
that validate `X` and optional `y` or `sample_weight`, prefer `check_inputs`;
it handles feature metadata, dtype conversion, array order, sparse support,
length checks, and fit-vs-inference validation consistently with scikit-learn.
Omit `y` or `sample_weight` when the method does not validate those inputs.

```python
from cuml.internals import mlfunc
from cuml.internals.validation import check_inputs, check_is_fitted


@mlfunc(set_input_type=True)
def fit(self, X, y):
    X, y = check_inputs(
        self,
        X,
        y,
        dtype=("float32", "float64"),
        order="K",
        reset=True,
    )
    rows, cols = X.shape
    dtype = X.dtype
    ...


@mlfunc(preserve_index=True)
def transform(self, X):
    check_is_fitted(self)
    X = check_inputs(
        self,
        X,
        dtype=self.result_.dtype,
        order="K",
    )
    ...
```

Use lower-level helpers directly when a method has non-standard inputs:
`check_array` for a standalone array, `check_y` for targets, `check_cudf` for
dataframe-oriented paths, and `check_all_finite` or `check_non_negative` for
specialized checks not already covered by the higher-level helpers.

### Returning Arrays

Return ``cupy`` or ``numpy`` arrays directly from reflected methods. The
reflection machinery will coerce these to the proper output type.

## Estimator Design

All `cuml.Base` estimators follow the [scikit-learn estimator
contract](https://scikit-learn.org/stable/developers/develop.html) plus the
cuML-specific rules below.

### Initialization

All estimators should accept `verbose` and `output_type`, and pass them to
`super().__init__()`.

Constructor parameters should be keyword-only unless the matched source API
uses positional parameters. This reduces breaking changes when parameters are
added or removed:

```python

# For an estimator that matches scikit-learn's API where the eps argument can
# be positional:
def __init__(self, eps=0.5, *, min_samples=5, max_mbytes_per_batch=None,
             calc_core_sample_indices=True, verbose=False, output_type=None):
    super().__init__(verbose=verbose, output_type=output_type)
    self.eps = eps
    self.min_samples = min_samples
    self.max_mbytes_per_batch = max_mbytes_per_batch
    self.calc_core_sample_indices = calc_core_sample_indices

# For an estimator that doesn't match any existing API:
def __init__(self, *, eps=0.5, min_samples=5, max_mbytes_per_batch=None,
             calc_core_sample_indices=True, verbose=False, output_type=None):
    super().__init__(verbose=verbose, output_type=output_type)
    self.eps = eps
    self.min_samples = min_samples
    self.max_mbytes_per_batch = max_mbytes_per_batch
    self.calc_core_sample_indices = calc_core_sample_indices
```

Store constructor arguments exactly as passed. Do not normalize, validate, or
convert them in `__init__`; doing so breaks cloning. See scikit-learn's
[instantiation
guidance](https://scikit-learn.org/stable/developers/develop.html#instantiation)
for details.

For example, the following `__init__` shows what **NOT** to do:

```python
def __init__(self, my_option="option1"):
   if (my_option == "option1"):
      self.my_option = 1
   else:
      self.my_option = 2
```

Instead, save `my_option` as-is and derive any normalized value later, usually
in `fit()` or a private helper.

### Implementing `_get_param_names()`

To support cloning, implement `_get_param_names()`. It should return
constructor parameter names, including names from `super()._get_param_names()`.
`Base.get_params()` reads these attributes and passes them to a new estimator
constructor, so every returned name must be accepted by `__init__()`.

```python
@classmethod
def _get_param_names(cls):
    return [
        *super()._get_param_names(),
        "eps",
        "min_samples",
    ]
```

Do not omit `super()._get_param_names()`; it includes base estimator parameters
such as `verbose` and `output_type`.

### Estimator Tags and cuML-Specific Tags

Estimator tags describe capabilities such as sparse support, positive-input
requirements, and preferred input layout. cuML supports the tags defined by the
scikit-learn estimator [developer
guide](https://scikit-learn.org/stable/developers/index.html), plus these
cuML-specific tags:

- `X_types_gpu` (default=['2darray'])
   Device-accessible input types accepted by the estimator.`2darray` includes
   CuPy, Numba device arrays, and cuDF objects. `sparse` includes CuPy sparse
   arrays.

- `preferred_input_order` (default=None)
   One of ['F', 'C', None]. Use `F` or `C` only when the estimator consistently
   benefits from that dense memory layout; otherwise leave it as `None`.

Use `__sklearn_tags__()` for estimator-specific tag overrides. The method
should call `super().__sklearn_tags__()`, mutate the returned structured `Tags`
object, and return it:

```python
def __sklearn_tags__(self):
   tags = super().__sklearn_tags__()
   tags.input_tags.positive_only = True
   return tags
```

Prefer the existing tag mixins for common capabilities. Tag-providing mixins
are cooperative: each mixin calls `super().__sklearn_tags__()` and mutates the
returned object. Put tag mixins to the left of `Base`, following scikit-learn's
[estimator MRO
guidance](https://scikit-learn.org/stable/developers/develop.html), so the
`super()` chain reaches the mixins before `Base` and `TagsMixin`:

```python
class MyEstimator(CMajorInputTagMixin, AllowNaNTagMixin, Base):
   ...
```

Do not add `_more_tags`, `_more_static_tags`, or `_get_tags` to new estimators.
Those are legacy scikit-learn tag APIs and can cause compatibility warnings or
fallback behavior in newer scikit-learn versions.

### Estimator Array-Like Attributes

Array-like fitted attributes should use `cuml.internals.ReflectedAttr` so
user-facing attribute reads respect cuML output-type settings.

Internally, a descriptor behaves like a normal attribute and returns the value
that was set. Externally, it lazily converts the value to the requested output
type and caches repeated conversions.

Lazy conversion reduces unnecessary memory use, but benchmarks must account for
the first attribute read if they need to include conversion cost.

#### Defining Array-Like Attributes

Declare descriptor-managed attributes as class variables.

```python
from cuml.internals import ReflectedAttr
from cuml.internals.base import Base

class TestEstimator(Base):

   # Class variables outside of any function
   my_cuml_array_ = ReflectedAttr()

   def __init__(self, ...):
      ...
```

#### Working with `ReflectedAttr`

Once a descriptor attribute is defined, use it like a normal attribute inside
estimator methods:

```python
import cupy as cp
from cuml.internals import mlfunc, ReflectedAttr
from cuml.internals.base import Base
from cuml.internals.validation import check_inputs

class SampleEstimator(Base):

   # Class variables outside of any function
   my_array_ = ReflectedAttr()
   my_other_array_ = ReflectedAttr()

   @mlfunc(set_input_type=True)
   def fit(self, X):
      # reset=True on check_inputs sets n_features_in_ and feature_names_in_
      X = check_inputs(self, X, order="K", reset=True)

      # Set descriptor-managed fitted attributes with validated arrays
      # When accessed in any `mlfunc`-decorated method, these will have the
      # same type they were originally set with.
      self.my_array_ = X
      self.my_other_array_ = cp.ones((10, 10))

      return self
```

Inside cuML code, descriptor attributes return the stored value. To
intentionally read one with a specific output type, use
`cuml.using_output_type()`:

```python
@mlfunc
def score(self):

   # Set the global output type to numpy
   with cuml.using_output_type("numpy"):
      # Accessing my_other_array_ will return a numpy array and
      # the result can be returned directly
      return np.sum(self.my_other_array_, axis=0)
```

This uses the same lazy conversion and caching path as external user reads.

#### `ReflectedAttr` External Functionality

Externally, descriptor attributes lazily convert to the active output type:

```python
my_est = SampleEstimator()

# Call fit() with a numpy array as the input
np_arr = np.ones((10,))
my_est.fit(np_arr) # This will load data into attributes

# Externally, descriptors reflect the fit-time input type by default
print(type(my_est.my_array_)) # Output: NumPy (saved from the input of `fit`)

# Calling fit again with cupy arrays, will have a similar effect
my_est.fit(cp.ones((10,)))
print(type(my_est.my_array_)) # Output: CuPy

# Setting the `output_type` will change all descriptor properties
# and ignore the input type
my_est.output_type = "cudf"

# Reading any of the attributes will convert the type lazily
print(type(my_est.my_array_)) # Output: cuDF object

# A global output type overrides the estimator output_type attribute
with cuml.using_output_type("cupy"):
  print(type(my_est.my_array_)) # Output: cupy

# Once the global output type is restored, we return to the estimator output_type
print(type(my_est.my_array_)) # Output: cuDF. Using a cached value!
```

### Estimator Methods

cuML uses reflection to convert public array outputs to the user's expected
type (`cupy`, `numpy`, `pandas`, `cudf`, etc.). Internal calls stay in an
internal context so intermediate computations avoid unnecessary conversions.

#### Using the `@mlfunc` Decorator

Use `@mlfunc` on methods that return arrays to the user:

```python
import cupy as cp
from cuml.internals import mlfunc, ReflectedAttr
from cuml.internals.base import Base
from cuml.internals.validation import check_inputs, check_is_fitted

class MyEstimator(Base):
    coef_ = ReflectedAttr()

    @mlfunc(set_input_type=True)
    def fit(self, X):
        self.coef_ = check_inputs(self, X, order="K", reset=True)
        return self

    @mlfunc(preserve_index=True)
    def predict(self, X):
        check_is_fitted(self)
        X = check_inputs(self, X, order="K")
        return X + cp.ones(X.shape)
```

| Decorator Usage | When to Use |
| :-------------- | :---------- |
| `@mlfunc(set_input_type=True)` | Fit-like methods that store `_input_type` through reflection while validation helpers set or check `n_features_in_`. |
| `@mlfunc(preserve_index=True)` | Transform/predict methods that return arrays with `n_samples` aligned with `X` |
| `@mlfunc(array_arg=None)` | Methods with no array input (e.g., `KernelDensity.sample()`). Uses fit-time input type. |

#### Handling Class Labels

Some methods (like classifier `predict` methods) need to return class labels,
which may include non-numeric dtypes. These are best handled via a custom
`ClassLabels` wrapper. This wrapper takes a cupy array of encoded label
indices, and a numpy array of the class labels corresponding to those indices.
The reflection machinery will lazily revert the encoding when it converts the
output to the requested `output_type`. If the output type doesn't support
the label dtype (e.g. `cupy` doesn't support `object` types), an informative
error will be raised.

```python
import cupy as cp
from cuml.internals.outputs import mlfunc, ClassLabels
from cuml.internals.base import Base

class MyClassifier(Base):
    @mlfunc(preserve_index=True)
    def predict(self, X):
        # Reflected methods return internal arrays inside this context.
        scores = self.decision_function(X)

        # Manual processing
        indices = (scores >= 0).view(cp.int8)

        return ClassLabels(indices, self.classes_)
```

#### Score Methods

Score methods typically return scalars. You may pass in `convert_output=False`
to avoid the output conversion machinery when it's known to not be needed.

```python
@mlfunc(convert_output=False)
def score(self, X, y):
    predictions = self.predict(X)
    return accuracy_score(y, predictions)
```

#### Property Accessors

Use `@mlfunc` on properties that return arrays:

```python
@property
@mlfunc
def support_(self):
    return self._support_vectors
```

#### Reflection Decision Rules

Use these rules when choosing a decorator:

- If a public method returns array-like data directly to the user, use
  `@mlfunc`.

- For fit-like methods, use `@mlfunc(set_input_type=True)` along with the
  appropriate validation helper functions.

- Inference methods should typically use `@mlfunc(preserve_index=True)` to
  ensure dataframe-like outputs have the same index as the input.

- If a method has no array input and should use the fit-time input type for
  output conversion, use `@mlfunc(array_arg=None)`.

- If a method returns a scalar or needs to call reflected methods internally
  without automatic output conversion, use `@mlfunc(convert_output=False)`.

When testing reflected methods, cover the default input-reflection behavior,
estimator-level `output_type`, and global `cuml.using_output_type(...)`
overrides.
