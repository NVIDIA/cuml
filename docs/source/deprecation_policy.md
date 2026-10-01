# Compatibility and deprecation policy

This policy describes which cuML APIs users can expect to remain compatible and
how incompatible changes are announced. Release-specific changes and migration
instructions are recorded in the [release notes](release_notes.md).

## Public Python APIs

Public Python APIs normally remain available for one release after they are
deprecated. Using a deprecated API during that transition raises a
`FutureWarning`. The warning and the API documentation identify the release in
which the deprecation was introduced, the release in which the API is expected
to be removed, and an alternative when one is available.

An explicitly announced longer transition remains authoritative. For example,
`cuml.tsa` was deprecated in 26.08 and is scheduled for removal in 26.12.

The Python deprecation policy does not apply to:

- names prefixed with `_`;
- APIs in namespaces designated as internal, including `cuml.internals`,
  `cuml.utils`, and `cuml.common`; or
- APIs in the `cuml.experimental` namespace.

Other APIs are considered public based on their presence in the public
documentation and examples, or an explicit declaration that they are public.

## C++ APIs

The documented C++ interfaces primarily support cuML's Python bindings and
internal implementation. They currently have no stability,
backward-compatibility, or deprecation guarantee and may change or be removed
without notice. Prefer the documented Python APIs when possible.

Contributors implementing a Python API deprecation should follow the
[developer deprecation policy](developer_guide/python/development.md),
which defines warning, documentation, and testing requirements.
