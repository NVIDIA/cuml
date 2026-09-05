/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "simple_mat.cuh"

#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda/std/functional>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <vector>

namespace ML {
namespace GLM {
namespace detail {

template <typename T>
inline void linearFwd(const raft::handle_t& handle,
                      SimpleDenseMat<T>& Z,
                      const SimpleMat<T>& X,
                      const SimpleDenseMat<T>& W)
{
  cudaStream_t stream = handle.get_stream().get();
  // Forward pass:  compute Z <- W * X.T + bias
  const bool has_bias = X.n != W.n;
  const int D         = X.n;
  if (has_bias) {
    SimpleVec<T> bias;
    SimpleDenseMat<T> weights;
    col_ref(W, bias, D);
    col_slice(W, weights, 0, D);
    // We implement Z <- W * X^T + b by
    // - Z <- b (broadcast): TODO reads Z unnecessarily atm
    // - Z <- W * X^T + Z    : TODO can be fused in CUTLASS?
    auto set_bias = [] __device__(const T z, const T b) { return b; };
    raft::linalg::matrixVectorOp<false, false>(
      Z.data, Z.data, bias.data, Z.n, Z.m, set_bias, stream);

    Z.assign_gemm(handle, 1, weights, false, X, true, 1, stream);
  } else {
    Z.assign_gemm(handle, 1, W, false, X, true, 0, stream);
  }
}

template <typename T>
inline void linearBwd(const raft::handle_t& handle,
                      SimpleDenseMat<T>& G,
                      const SimpleMat<T>& X,
                      const SimpleDenseMat<T>& dZ,
                      bool setZero)
{
  cudaStream_t stream = handle.get_stream().get();
  // Backward pass:
  // - compute G <- dZ * X.T
  // - for bias: Gb = mean(dZ, 1)

  const bool has_bias = X.n != G.n;
  const int D         = X.n;
  const T beta        = setZero ? T(0) : T(1);
  if (has_bias) {
    SimpleVec<T> Gbias;
    SimpleDenseMat<T> Gweights;
    col_ref(G, Gbias, D);
    col_slice(G, Gweights, 0, D);

    // TODO can this be fused somehow?
    Gweights.assign_gemm(handle, 1.0 / X.m, dZ, false, X, false, beta, stream);
    raft::stats::mean<true>(Gbias.data, dZ.data, dZ.m, dZ.n, false, stream);
  } else {
    G.assign_gemm(handle, 1.0 / X.m, dZ, false, X, false, beta, stream);
  }
}

struct GLMDims {
  bool fit_intercept;
  int C, D, dims, n_param;
  GLMDims(int C, int D, bool fit_intercept) : C(C), D(D), fit_intercept(fit_intercept)
  {
    dims    = D + fit_intercept;
    n_param = dims * C;
  }
};

template <typename T, class Loss>
struct GLMBase : GLMDims {
  typedef SimpleDenseMat<T> Mat;
  typedef SimpleVec<T> Vec;

  const raft::handle_t& handle;
  T* sample_weights;
  T weights_sum;

  GLMBase(const raft::handle_t& handle, int D, int C, bool fit_intercept)
    : GLMDims(C, D, fit_intercept), handle(handle), sample_weights(nullptr), weights_sum(0)
  {
  }

  void add_sample_weights(T* sample_weights, int n_samples, cudaStream_t stream)
  {
    this->sample_weights = sample_weights;
    this->weights_sum    = thrust::reduce(thrust::cuda::par.on(stream),
                                       sample_weights,
                                       sample_weights + n_samples,
                                       (T)0,
                                       cuda::std::plus<T>());
  }

  /*
   * Computes the following:
   * 1. Z <- dL/DZ
   * 2. loss_val <- sum loss(Z)
   *
   * Default: elementwise application of loss and its derivative
   *
   * NB: for this method to work, loss implementations must have two functor fields `lz` and `dlz`.
   *     These two compute loss value and its derivative w.r.t. `z`.
   */
  inline void getLossAndDZ(T* loss_val,
                           SimpleDenseMat<T>& Z,
                           const SimpleVec<T>& y,
                           cudaStream_t stream)
  {
    // Base impl assumes simple case C = 1
    // TODO would be nice to have a kernel that fuses these two steps
    // This would be easy, if mapThenSumReduce allowed outputting the result of
    // map (supporting inplace)
    auto lz_copy  = static_cast<Loss*>(this)->lz;
    auto dlz_copy = static_cast<Loss*>(this)->dlz;
    if (this->sample_weights) {  // Sample weights are in use
      T normalization = 1.0 / this->weights_sum;
      raft::linalg::mapThenSumReduce(
        loss_val,
        y.len,
        [lz_copy, normalization] __device__(const T y, const T z, const T weight) {
          return lz_copy(y, z) * (weight * normalization);
        },
        stream,
        y.data,
        Z.data,
        sample_weights);
      raft::linalg::map_k(
        Z.data,
        y.len,
        [dlz_copy] __device__(const T y, const T z, const T weight) {
          return weight * dlz_copy(y, z);
        },
        stream,
        y.data,
        Z.data,
        sample_weights);
    } else {  // Sample weights are not used
      T normalization = 1.0 / y.len;
      raft::linalg::mapThenSumReduce(
        loss_val,
        y.len,
        [lz_copy, normalization] __device__(const T y, const T z) {
          return lz_copy(y, z) * normalization;
        },
        stream,
        y.data,
        Z.data);
      raft::linalg::binaryOp(Z.data, y.data, Z.data, y.len, dlz_copy, stream);
    }
  }

  inline void loss_grad(T* loss_val,
                        Mat& G,
                        const Mat& W,
                        const SimpleMat<T>& Xb,
                        const Vec& yb,
                        Mat& Zb,
                        cudaStream_t stream,
                        bool initGradZero = true)
  {
    Loss* loss = static_cast<Loss*>(this);  // static polymorphism

    linearFwd(handle, Zb, Xb, W);                  // linear part: forward pass
    loss->getLossAndDZ(loss_val, Zb, yb, stream);  // loss specific part
    linearBwd(handle, G, Xb, Zb, initGradZero);    // linear part: backward pass
  }
};

template <typename T, class GLMObjective>
struct GLMWithData : GLMDims {
  const SimpleMat<T>* X;
  const SimpleVec<T>* y;
  SimpleDenseMat<T>* Z;
  GLMObjective* objective;

  GLMWithData(GLMObjective* obj, const SimpleMat<T>& X, const SimpleVec<T>& y, SimpleDenseMat<T>& Z)
    : objective(obj), X(&X), y(&y), Z(&Z), GLMDims(obj->C, obj->D, obj->fit_intercept)
  {
  }

  // interface exposed to typical non-linear optimizers
  inline T operator()(const SimpleVec<T>& wFlat,
                      SimpleVec<T>& gradFlat,
                      T* dev_scalar,
                      cudaStream_t stream)
  {
    SimpleDenseMat<T> W(wFlat.data, C, dims);
    SimpleDenseMat<T> G(gradFlat.data, C, dims);
    objective->loss_grad(dev_scalar, G, W, *X, *y, *Z, stream);
    T loss_host;
    raft::update_host(&loss_host, dev_scalar, 1, stream);
    raft::interruptible::synchronize(stream);
    return loss_host;
  }

  /**
   * @brief Calculate a norm of the gradient computed using the given Loss instance.
   *
   * This function is intended to be used in `check_convergence`; it's output is supposed
   * to be proportional to the loss value w.r.t. the number of features (D).
   *
   * Different loss functions may scale differently with the number of features (D).
   * This has an effect on the convergence criteria. To account for that, we let a
   * loss function define its preferred metric. Normally, we differentiate between the
   * L2 norm (e.g. for Squared loss) and LInf norm (e.g. for Softmax loss).
   */
  inline T gradNorm(const SimpleVec<T>& grad, T* dev_scalar, cudaStream_t stream)
  {
    return objective->gradNorm(grad, dev_scalar, stream);
  }
};
};  // namespace detail
};  // namespace GLM
};  // namespace ML
