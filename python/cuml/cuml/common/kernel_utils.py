#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import functools
import uuid

import cupy as cp

import cuml.internals.logger as logger

__all__ = ("cuda_kernel_factory",)

# A mapping of dtype -> kernel type
# See https://docs.cupy.dev/en/latest/user_guide/kernel.html#kernel-arguments
_dtype_str_map = {
    cp.dtype("bool"): "bool",
    cp.dtype("int8"): "char",
    cp.dtype("int16"): "short",
    cp.dtype("int32"): "int",
    cp.dtype("int64"): "long long",
    cp.dtype("uint8"): "unsigned char",
    cp.dtype("uint16"): "unsigned short",
    cp.dtype("uint32"): "unsigned int",
    cp.dtype("uint64"): "unsigned long long",
    cp.dtype("float16"): "__half",
    cp.dtype("float32"): "float",
    cp.dtype("float64"): "double",
}

# Check if bfloat16 support is installed
try:
    bfloat16 = cp.dtype("bfloat16")
except TypeError:
    pass
else:
    _dtype_str_map[bfloat16] = "__nv_bfloat16"


@functools.lru_cache(maxsize=5000)
def cuda_kernel_factory(source, dtypes, kernel_name=None):
    """
    A factory wrapper function to perform some of the boiler-plate involved in
    making cuPy RawKernels type-agnostic.

    This function will perform a string search and replace of c-based datatype
    primitives in ``source`` using a numerical placeholder (eg. {0}, {1}) for the
    dtype in the corresponding index of tuple ``dtypes``.

    Note that the extern, function scope, and function name should not be
    included in the kernel string. These will be added by this function and
    the function name will be made unique, based on the given dtypes.

    Parameters
    ----------
    source : str
        Valid nvrtc kernel source string without extern, scope, or function name.
    dtypes : Iterable[dtype]
        An iterable of dtypes to search and replace.
    kernel_name : str
        The kernel name to use. If not provided, a unique name will be
        generated, which will also prevent this function from being memoized.

    Returns
    -------
    raw_kernel : cupy.RawKernel object ready for use
    """
    includes = set()
    dtypes = tuple(cp.dtype(dtype) for dtype in dtypes)
    dtype_strs = []

    for idx, dtype in enumerate(dtypes):
        if (dtype_str := _dtype_str_map.get(dtype)) is None:
            raise ValueError(f"{dtype} is not a valid type for this kernel.")

        if dtype == "float16":
            includes.add("#include <cuda_fp16.h>\n")
        elif dtype == "bfloat16":
            includes.add("#include <cuda_bf16.h>\n")

        dtype_strs.append(dtype_str)
        source = source.replace(f"{{{idx}}}", dtype_str)

    if kernel_name is None:
        kernel_name = f"kernel_{uuid.uuid4().hex}"

    logger.debug(f"Compiling `{kernel_name}` for {tuple(map(str, dtypes))}")

    suffix = "".join(dtype_strs).replace(" ", "_")
    kernel_name = f"{kernel_name}_{suffix}"
    include_str = "".join(sorted(includes))

    source = f'{include_str}extern "C" __global__\nvoid {kernel_name}{source}'
    return cp.RawKernel(source, kernel_name)
