/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/handle.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/linalg/strided_reduction.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/memory.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cuml {
namespace genetic {

template <typename math_t = float>
void weightedPearson(const raft::handle_t& h,
                     const uint64_t n_samples,
                     const uint64_t n_progs,
                     const math_t* Y,
                     const math_t* X,
                     const math_t* W,
                     math_t* out)
{
  // Find Pearson's correlation coefficient

  cudaStream_t stream = h.get_stream().get();

  rmm::device_uvector<math_t> corr(n_samples * n_progs, stream);

  rmm::device_uvector<math_t> y_tmp(n_samples, stream);
  rmm::device_uvector<math_t> x_tmp(n_samples * n_progs, stream);

  rmm::device_scalar<math_t> y_mu(stream);            // output mean
  rmm::device_uvector<math_t> x_mu(n_progs, stream);  // predicted output mean

  rmm::device_uvector<math_t> y_diff(n_samples, stream);  // normalized output
  rmm::device_uvector<math_t> x_diff(n_samples * n_progs,
                                     stream);  // normalized predicted output

  rmm::device_uvector<math_t> y_std(1, stream);  // output stddev
  rmm::device_uvector<math_t> x_std(n_progs,
                                    stream);  // predicted output stddev

  rmm::device_scalar<math_t> dWS(stream);  // sample weight sum
  math_t N = (math_t)n_samples;

  // Sum of weights
  raft::stats::sum<false>(dWS.data(), W, (uint64_t)1, n_samples, stream);
  math_t WS = dWS.value(stream);

  // Find y_mu
  raft::linalg::matrixVectorOp<false, false>(
    y_tmp.data(),
    Y,
    W,
    (uint64_t)1,
    n_samples,
    [N, WS] __device__(math_t y, math_t w) { return N * w * y / WS; },
    stream);

  raft::stats::mean<false>(y_mu.data(), y_tmp.data(), (uint64_t)1, n_samples, false, stream);

  // Find x_mu
  raft::linalg::matrixVectorOp<false, true>(
    x_tmp.data(),
    X,
    W,
    n_progs,
    n_samples,
    [N, WS] __device__(math_t x, math_t w) { return N * w * x / WS; },
    stream);

  raft::stats::mean<false>(x_mu.data(), x_tmp.data(), n_progs, n_samples, false, stream);

  // Find y_diff
  raft::stats::meanCenter<false, true>(
    y_diff.data(), Y, y_mu.data(), (uint64_t)1, n_samples, stream);

  // Find x_diff
  raft::stats::meanCenter<false, true>(x_diff.data(), X, x_mu.data(), n_progs, n_samples, stream);

  // Find y_std
  raft::linalg::stridedReduction(
    y_std.data(),
    y_diff.data(),
    (uint64_t)1,
    n_samples,
    (math_t)0,
    stream,
    false,
    [W] __device__(math_t v, int i) { return v * v * W[i]; },
    raft::add_op(),
    [] __device__(math_t in) { return raft::sqrt(in); });
  math_t HYstd = y_std.element(0, stream);

  // Find x_std
  raft::linalg::stridedReduction(
    x_std.data(),
    x_diff.data(),
    n_progs,
    n_samples,
    (math_t)0,
    stream,
    false,
    [W] __device__(math_t v, int i) { return v * v * W[i]; },
    raft::add_op(),
    [] __device__(math_t in) { return raft::sqrt(in); });

  // Cross covariance
  raft::linalg::matrixVectorOp<false, false>(
    corr.data(),
    x_diff.data(),
    y_diff.data(),
    W,
    n_progs,
    n_samples,
    [N, HYstd] __device__(math_t xd, math_t yd, math_t w) { return N * w * xd * yd / HYstd; },
    stream);

  // Find Correlation coeff
  raft::linalg::matrixVectorOp<false, true>(
    corr.data(),
    corr.data(),
    x_std.data(),
    n_progs,
    n_samples,
    [] __device__(math_t c, math_t xd) { return c / xd; },
    stream);

  raft::stats::mean<false>(out, corr.data(), n_progs, n_samples, false, stream);
}

struct rank_functor {
  template <typename math_t>
  __host__ __device__ math_t operator()(math_t data)
  {
    if (data == 0)
      return 0;
    else
      return 1;
  }
};

template <typename math_t = float>
void weightedSpearman(const raft::handle_t& h,
                      const uint64_t n_samples,
                      const uint64_t n_progs,
                      const math_t* Y,
                      const math_t* Y_pred,
                      const math_t* W,
                      math_t* out)
{
  cudaStream_t stream = h.get_stream().get();

  // Get ranks for Y
  thrust::device_vector<math_t> Ycopy(Y, Y + n_samples);
  thrust::device_vector<math_t> rank_idx(n_samples, 0);
  thrust::device_vector<math_t> rank_diff(n_samples, 0);
  thrust::device_vector<math_t> Yrank(n_samples, 0);

  auto exec_policy = rmm::exec_policy(stream);

  thrust::sequence(exec_policy, rank_idx.begin(), rank_idx.end(), 0);
  thrust::sort_by_key(exec_policy, Ycopy.begin(), Ycopy.end(), rank_idx.begin());
  thrust::adjacent_difference(exec_policy, Ycopy.begin(), Ycopy.end(), rank_diff.begin());
  thrust::transform(
    exec_policy, rank_diff.begin(), rank_diff.end(), rank_diff.begin(), rank_functor());
  rank_diff[0] = 1;
  thrust::inclusive_scan(exec_policy, rank_diff.begin(), rank_diff.end(), rank_diff.begin());
  thrust::copy(rank_diff.begin(),
               rank_diff.end(),
               thrust::make_permutation_iterator(Yrank.begin(), rank_idx.begin()));

  // Get ranks for Y_pred
  // TODO: Find a better way to batch this
  thrust::device_vector<math_t> Ypredcopy(Y_pred, Y_pred + n_samples * n_progs);
  thrust::device_vector<math_t> Ypredrank(n_samples * n_progs, 0);
  thrust::device_ptr<math_t> Ypredptr     = thrust::device_pointer_cast<math_t>(Ypredcopy.data());
  thrust::device_ptr<math_t> Ypredrankptr = thrust::device_pointer_cast<math_t>(Ypredrank.data());

  for (std::size_t i = 0; i < n_progs; ++i) {
    thrust::sequence(exec_policy, rank_idx.begin(), rank_idx.end(), 0);
    thrust::sort_by_key(
      exec_policy, Ypredptr + (i * n_samples), Ypredptr + ((i + 1) * n_samples), rank_idx.begin());
    thrust::adjacent_difference(
      exec_policy, Ypredptr + (i * n_samples), Ypredptr + ((i + 1) * n_samples), rank_diff.begin());
    thrust::transform(
      exec_policy, rank_diff.begin(), rank_diff.end(), rank_diff.begin(), rank_functor());
    rank_diff[0] = 1;
    thrust::inclusive_scan(exec_policy, rank_diff.begin(), rank_diff.end(), rank_diff.begin());
    thrust::copy(
      rank_diff.begin(),
      rank_diff.end(),
      thrust::make_permutation_iterator(Ypredrankptr + (i * n_samples), rank_idx.begin()));
  }

  // Compute pearson's coefficient
  weightedPearson(h,
                  n_samples,
                  n_progs,
                  thrust::raw_pointer_cast(Yrank.data()),
                  thrust::raw_pointer_cast(Ypredrank.data()),
                  W,
                  out);
}

template <typename math_t = float>
void meanAbsoluteError(const raft::handle_t& h,
                       const uint64_t n_samples,
                       const uint64_t n_progs,
                       const math_t* Y,
                       const math_t* Y_pred,
                       const math_t* W,
                       math_t* out)
{
  cudaStream_t stream = h.get_stream().get();
  rmm::device_uvector<math_t> error(n_samples * n_progs, stream);
  rmm::device_scalar<math_t> dWS(stream);
  math_t N = (math_t)n_samples;

  // Weight Sum
  raft::stats::sum<false>(dWS.data(), W, (uint64_t)1, n_samples, stream);
  math_t WS = dWS.value(stream);

  // Compute absolute differences
  raft::linalg::matrixVectorOp<false, false>(
    error.data(),
    Y_pred,
    Y,
    W,
    n_progs,
    n_samples,
    [N, WS] __device__(math_t y_p, math_t y, math_t w) { return N * w * raft::abs(y - y_p) / WS; },
    stream);

  // Average along rows
  raft::stats::mean<false>(out, error.data(), n_progs, n_samples, false, stream);
}

template <typename math_t = float>
void meanSquareError(const raft::handle_t& h,
                     const uint64_t n_samples,
                     const uint64_t n_progs,
                     const math_t* Y,
                     const math_t* Y_pred,
                     const math_t* W,
                     math_t* out)
{
  cudaStream_t stream = h.get_stream().get();
  rmm::device_uvector<math_t> error(n_samples * n_progs, stream);
  rmm::device_scalar<math_t> dWS(stream);
  math_t N = (math_t)n_samples;

  // Weight Sum
  raft::stats::sum<false>(dWS.data(), W, (uint64_t)1, n_samples, stream);
  math_t WS = dWS.value(stream);

  // Compute square differences
  raft::linalg::matrixVectorOp<false, false>(
    error.data(),
    Y_pred,
    Y,
    W,
    n_progs,
    n_samples,
    [N, WS] __device__(math_t y_p, math_t y, math_t w) {
      return N * w * (y_p - y) * (y_p - y) / WS;
    },
    stream);

  // Add up row values per column
  raft::stats::mean<false>(out, error.data(), n_progs, n_samples, false, stream);
}

template <typename math_t = float>
void rootMeanSquareError(const raft::handle_t& h,
                         const uint64_t n_samples,
                         const uint64_t n_progs,
                         const math_t* Y,
                         const math_t* Y_pred,
                         const math_t* W,
                         math_t* out)
{
  cudaStream_t stream = h.get_stream().get();

  // Find MSE
  meanSquareError(h, n_samples, n_progs, Y, Y_pred, W, out);

  // Take sqrt on all entries
  raft::linalg::sqrt(out, out, n_progs, stream);
}

template <typename math_t = float>
void logLoss(const raft::handle_t& h,
             const uint64_t n_samples,
             const uint64_t n_progs,
             const math_t* Y,
             const math_t* Y_pred,
             const math_t* W,
             math_t* out)
{
  cudaStream_t stream = h.get_stream().get();
  // Logistic error per sample
  rmm::device_uvector<math_t> error(n_samples * n_progs, stream);
  rmm::device_scalar<math_t> dWS(stream);
  math_t N = (math_t)n_samples;

  // Weight Sum
  raft::stats::sum<false>(dWS.data(), W, (uint64_t)1, n_samples, stream);
  math_t WS = dWS.value(stream);

  // Compute logistic loss as described in
  // http://fa.bianp.net/blog/2019/evaluate_logistic/
  // in an attempt to avoid encountering nan values. Modified for weighted logistic regression.
  raft::linalg::matrixVectorOp<false, false>(
    error.data(),
    Y_pred,
    Y,
    W,
    n_progs,
    n_samples,
    [N, WS] __device__(math_t yp, math_t y, math_t w) {
      math_t logsig;
      if (yp < -33.3)
        logsig = yp;
      else if (yp <= -18)
        logsig = yp - expf(yp);
      else if (yp <= 37)
        logsig = -log1pf(expf(-yp));
      else
        logsig = -expf(-yp);

      return ((1 - y) * yp - logsig) * (N * w / WS);
    },
    stream);

  // Take average along rows
  raft::stats::mean<false>(out, error.data(), n_progs, n_samples, false, stream);
}

}  // namespace genetic
}  // namespace cuml
