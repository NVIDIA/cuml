/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Python.h>
#include <cuml/common/callback.hpp>

#include <iostream>

namespace ML {
namespace Internals {

class DefaultGraphBasedDimRedCallback : public GraphBasedDimRedCallback {
 public:
  PyObject* get_cupy_array(void* embeddings)
  {
    PyObject* pycl = (PyObject*)this->pyCallbackClass;

    if (isFloat) {
      return PyObject_CallMethod(
        pycl, "get_cupy_array", "(llls)", embeddings, n, n_components, "float32");
    } else {
      return PyObject_CallMethod(
        pycl, "get_cupy_array", "(llls)", embeddings, n, n_components, "float64");
    }
  }

  void on_preprocess_end(void* embeddings) override
  {
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* array = get_cupy_array(embeddings);
    PyObject* res =
      PyObject_CallMethod(this->pyCallbackClass, "on_preprocess_end", "(O)", array);
    Py_DECREF(array);
    Py_DECREF(res);
    PyGILState_Release(gstate);
  }

  void on_epoch_end(void* embeddings) override
  {
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* array = get_cupy_array(embeddings);
    PyObject* res = PyObject_CallMethod(this->pyCallbackClass, "on_epoch_end", "(O)", array);
    Py_DECREF(array);
    Py_DECREF(res);
    PyGILState_Release(gstate);
  }

  void on_train_end(void* embeddings) override
  {
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* array = get_cupy_array(embeddings);
    PyObject* res = PyObject_CallMethod(this->pyCallbackClass, "on_train_end", "(O)", array);
    Py_DECREF(array);
    Py_DECREF(res);
    PyGILState_Release(gstate);
  }

 public:
  PyObject* pyCallbackClass = nullptr;
};

}  // namespace Internals
}  // namespace ML
