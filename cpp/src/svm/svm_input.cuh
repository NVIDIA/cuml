/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/matrix/kernel_params.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>

#include <cstdint>
#include <limits>

namespace ML {
namespace SVM {

template <typename math_t>
auto makeDenseSvmInputView(math_t* input,
                           int n_rows,
                           int n_cols,
                           ML::matrix::KernelType kernel_type)
{
  ASSERT(kernel_type != ML::matrix::KernelType::PRECOMPUTED ||
           std::int64_t{n_rows} * n_cols <= std::numeric_limits<int>::max(),
         "Precomputed SVM kernel matrices must have at most %d elements.",
         std::numeric_limits<int>::max());
  return raft::make_device_strided_matrix_view<math_t, int, raft::layout_f_contiguous>(
    input, n_rows, n_cols, 0);
}

}  // namespace SVM
}  // namespace ML
