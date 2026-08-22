#pragma once

/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/export.hpp>

namespace CUML_EXPORT MLCommon {
namespace Matrix {

    /**
     * @brief This is a *helper* wrapper around the multi-gpu data blocks owned
     * by a worker. It's design is NOT final. Its so written this way to get
     * something concrete in a short span of time.
     * @todo add support for custom allocators
     */
    template <typename Type>
    struct Data {
      Data() : ptr(nullptr), nElements(0), totalSize(0) {}
      Data(Type* _ptr, size_t _n_elements)
        : ptr(_ptr),
          nElements(_n_elements),
          totalSize(ML::checked_mul<size_t>(_n_elements, sizeof(Type)))
      {
      }

      void setNumElements(size_t _n_elements)
      {
        nElements = _n_elements;
        totalSize = ML::checked_mul<size_t>(_n_elements, sizeof(Type));
      }

      /**
       * actual data block. This is just a linearly laid out buffer of all blocks
       * owned by this worker
       */
      Type* ptr = nullptr;

      /**
       * number of elements in this buffer.
       */
      size_t nElements = 0;

      /**
       * total size (in bytes) of this buffer. In future, this will be passed
       * to the dealloc function underneath
       */
      size_t totalSize = (size_t)0;

      /**
       * Return the number of elements of Type in ptr.
       */
      size_t numElements() const { return nElements; }
    };

typedef Data<float> floatData_t;
typedef Data<double> doubleData_t;

};  // end namespace Matrix
};  // namespace CUML_EXPORT MLCommon
