#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cupy as cp

from libc.stdint cimport uintptr_t


cdef extern from "Python.h":
    cdef cppclass PyObject


cdef extern from "callbacks_implems.h" namespace "ML::Internals":
    cdef cppclass Callback:
        pass

    cdef cppclass DefaultGraphBasedDimRedCallback(Callback):
        void setup(int n, int d) except +
        void on_preprocess_end(void* embeddings) except +
        void on_epoch_end(void* embeddings) except +
        void on_train_end(void* embeddings) except +
        PyObject* pyCallbackClass


class _CuPyArrayWithCopyToHostShim(cp.ndarray):
    """A small shim to support `copy_to_host` on `cupy.ndarray` objects for a
    deprecation period."""

    def copy_to_host(self):
        """Deprecated. Perhaps you want `.get()` instead."""
        warnings.warn(
            "The `copy_to_host` method is deprecated and will be removed "
            "in version 27.02. The UMAP callback routines now take cupy arrays "
            "as arguments, please update your code accordingly",
            FutureWarning,
        )
        return self.get()


cdef class PyCallback:
    def get_cupy_array(self, ptr, n_rows, n_cols, typestr):
        dtype = cp.dtype(typestr)
        mem = cp.cuda.UnownedMemory(
            ptr=ptr, size=n_rows * n_cols * dtype.itemsize, owner=None
        )
        return _CuPyArrayWithCopyToHostShim(
            memptr=cp.cuda.memory.MemoryPointer(mem, 0),
            shape=(n_rows, n_cols),
            dtype=dtype,
            order="C",
        )


cdef class GraphBasedDimRedCallback(PyCallback):
    """
    Usage
    -----

    class CustomCallback(GraphBasedDimRedCallback):
        def on_preprocess_end(self, embeddings):
            print(embeddings)

        def on_epoch_end(self, embeddings):
            print(embeddings)

        def on_train_end(self, embeddings):
            print(embeddings)

    reducer = UMAP(n_components=2, callback=CustomCallback())
    """

    cdef DefaultGraphBasedDimRedCallback native_callback

    def __cinit__(self):
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def __reduce__(self):
        return (type(self), ())

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)

    def on_preprocess_end(self, embeddings):
        pass

    def on_epoch_end(self, embeddings):
        pass

    def on_train_end(self, embeddings):
        pass
