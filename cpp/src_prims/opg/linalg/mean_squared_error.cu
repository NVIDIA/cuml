/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/prims/opg/comm_utils.h>
#include <cuml/prims/opg/linalg/mean_squared_error.hpp>

#include <raft/linalg/mean_squared_error.cuh>

namespace MLCommon {
namespace LinAlg {
namespace opg {

// TODO: Once RAFT side is fixed, this needs to call RAFT's version directly
// ref: https://github.com/NVIDIA/raft/issues/872
template <typename math_t>
void meanSquaredError(
  math_t* out, const math_t* A, const math_t* B, size_t len, math_t weight, cudaStream_t stream)
{
  auto sq_diff = [len, weight] __device__(const math_t a, const math_t b) {
    math_t diff = a - b;
    return diff * diff * weight / len;
  };
  raft::linalg::mapThenSumReduce<math_t, decltype(sq_diff)>(out, len, sq_diff, stream, A, B);
}

template <typename math_t, int TPB = 256>
void meanSquaredError_impl(math_t* out,
                           const Matrix::Data<math_t>& in1,
                           const Matrix::PartDescriptor& in1Desc,
                           const Matrix::Data<math_t>& in2,
                           const Matrix::PartDescriptor& in2Desc,
                           const raft::comms::comms_t& comm,
                           cudaStream_t stream,
                           int root,
                           bool broadcastResult)
{
  ASSERT(in1Desc == in2Desc, "opg::meanSquaredError: in1/in2 descriptors must match!");
  ASSERT(in1Desc.layout == Matrix::Layout::LayoutRowMajor,
         "opg::meanSquaredError: currently only row major is supported!");
  size_t len = in1.totalSize / sizeof(math_t);

  // this weight corrects the mean computation on local ranks
  math_t w = (math_t)len / (in1Desc.M * in1Desc.N);
  RAFT_CUDA_TRY(cudaMemsetAsync(out, 0, sizeof(math_t), stream));

  if (len <= 0) { return; }

  meanSquaredError<math_t>(out, in1.ptr, in2.ptr, len, w, stream);

  if (broadcastResult) {
    MLCommon::opg::allreduce_single_sum(out, out, comm, stream);
  } else {
    MLCommon::opg::reduce_single_sum(out, out, comm, stream, root);
  }
}

/// Instantiations

void meanSquaredError(double* out,
                      const Matrix::Data<double>& in1,
                      const Matrix::PartDescriptor& in1Desc,
                      const Matrix::Data<double>& in2,
                      const Matrix::PartDescriptor& in2Desc,
                      const raft::comms::comms_t& comm,
                      cudaStream_t stream,
                      int root,
                      bool broadcastResult)
{
  meanSquaredError_impl<double>(
    out, in1, in1Desc, in2, in2Desc, comm, stream, root, broadcastResult);
}

void meanSquaredError(float* out,
                      const Matrix::Data<float>& in1,
                      const Matrix::PartDescriptor& in1Desc,
                      const Matrix::Data<float>& in2,
                      const Matrix::PartDescriptor& in2Desc,
                      const raft::comms::comms_t& comm,
                      cudaStream_t stream,
                      int root,
                      bool broadcastResult)
{
  meanSquaredError_impl<float>(
    out, in1, in1Desc, in2, in2Desc, comm, stream, root, broadcastResult);
}

};  // end namespace opg
};  // end namespace LinAlg
};  // end namespace MLCommon
