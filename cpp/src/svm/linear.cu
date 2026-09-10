/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <common/nvtx.hpp>

#include <cuml/common/checked_arithmetic.hpp>
#include <cuml/common/export.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/linear_model/glm.hpp>
#include <cuml/svm/linear.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/functional>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <omp.h>

namespace ML {
namespace SVM {
namespace linear {

inline int narrowDown(std::size_t n)
{
  ASSERT(std::size_t(std::numeric_limits<int>::max()) >= n,
         "LinearSVM supports input sizes only within `int` range at this point (got = %zu)",
         n);
  return int(n);
}

/** The loss function is the main hint for whether we solve classification or regression. */
inline bool isRegression(Params::Loss loss)
{
  return loss == Params::EPSILON_INSENSITIVE || loss == Params::SQUARED_EPSILON_INSENSITIVE;
}

/** A functor that maps the multiclass problem onto the one-vs-rest binary problem */
template <typename T>
struct OvrSelector {
  const T* classes;
  const int selected;
  __device__ T operator()(const T x) const { return x == classes[selected] ? 1 : 0; }
};

/**
 * The linear part of the prediction.
 *
 * @param [in] handle - raft handle
 * @param [in] X - column-major matrix of size (nRows, nCols)
 * @param [in] w - row-major matrix of size [nCols + fitIntercept, coefCols]
 * @param [in] nRows - number of samples
 * @param [in] nCols - number of features
 * @param [in] coefCols - number of columns in `w` (`nClasses == 2 ? 1 : nClasses`)
 * @param [in] fitIntercept - whether to add the bias term
 * @param [out] out - row-major matrix of size [nRows, coefCols]
 * @param [in] stream - cuda stream (not synchronized)
 */
template <typename T>
void predictLinear(const raft::handle_t& handle,
                   const T* X,
                   const T* w,
                   const std::size_t nRows,
                   const std::size_t nCols,
                   const std::size_t coefCols,
                   const bool fitIntercept,
                   T* out,
                   cudaStream_t stream)
{
  raft::linalg::gemm<T>(handle,
                        out,
                        (T*)X,
                        (T*)w,
                        narrowDown(nRows),
                        narrowDown(coefCols),
                        narrowDown(nCols),
                        false,
                        true,
                        false,
                        stream);

  if (fitIntercept)
    raft::linalg::matrixVectorOp<true, true>(
      out, out, w + nCols * coefCols, coefCols, nRows, cuda::std::plus{}, stream);
}

/** A helper struct for selecting handle/stream depending on whether omp parallel is active. */
class WorkerHandle {
 private:
  raft::handle_t* handle_ptr = nullptr;

 public:
  int stream_id = 0;
  const raft::handle_t& handle;
  cudaStream_t stream;

  WorkerHandle(const raft::handle_t& handle, cudaStream_t stream) : handle(handle), stream(stream)
  {
  }

  WorkerHandle(const raft::handle_t& h, int stream_id)
    : handle_ptr{new raft::handle_t{h.get_next_usable_stream(stream_id)}},
      stream_id(stream_id),
      handle(*handle_ptr),
      stream(h.get_next_usable_stream(stream_id).get())
  {
  }

  ~WorkerHandle()
  {
    if (handle_ptr != nullptr) delete handle_ptr;
  }
};

template <typename T>
int fit(const raft::handle_t& handle,
        const Params& params,
        const std::size_t nRows,
        const std::size_t nCols,
        const int nClasses,
        const T* classes,
        const T* X,
        const T* y,
        const T* sampleWeight,
        T* w)
{
  if (isRegression(params.loss)) {
    ASSERT(nClasses == 0 && classes == nullptr, "Regression fit takes no classes");
  } else {
    ASSERT(nClasses > 1 && classes != nullptr, "Must have > 1 class for classification");
  }

  cudaStream_t stream = handle.get_stream().get();

  const int coefCols         = nClasses <= 2 ? 1 : nClasses;
  const std::size_t coefRows = nCols + int(params.fit_intercept);

  raft::common::nvtx::range fun_scope("ML::SVM::linear::fit");

  auto nCols1 = nCols + int(params.fit_intercept && params.penalized_intercept);
  T iC        = params.C > 0 ? (1.0 / params.C) : 1.0;

  T* X1 = (T*)X;
  rmm::device_uvector<T> X1Buf(0, stream);
  if (params.fit_intercept && params.penalized_intercept) {
    std::size_t const x1_count = checked_mul<std::size_t>(nCols1, nRows);
    std::size_t const x_count  = checked_mul<std::size_t>(nCols, nRows);
    X1Buf.resize(x1_count, stream);
    X1 = X1Buf.data();
    raft::copy(X1, X, x_count, stream);
    thrust::device_ptr<T> p(X1 + x_count);
    thrust::fill(thrust::cuda::par.on(stream), p, p + nRows, 1.0);
  }

  ML::GLM::qn_params qn_pams;
  switch (params.loss) {
    case Params::HINGE: qn_pams.loss = ML::GLM::QN_LOSS_SVC_L1; break;
    case Params::SQUARED_HINGE: qn_pams.loss = ML::GLM::QN_LOSS_SVC_L2; break;
    case Params::EPSILON_INSENSITIVE: qn_pams.loss = ML::GLM::QN_LOSS_SVR_L1; break;
    case Params::SQUARED_EPSILON_INSENSITIVE: qn_pams.loss = ML::GLM::QN_LOSS_SVR_L2; break;
    default: break;
  }
  qn_pams.fit_intercept       = params.fit_intercept && !params.penalized_intercept;
  qn_pams.penalty_l1          = params.penalty == Params::L1 ? iC : 0.0;
  qn_pams.penalty_l2          = params.penalty == Params::L2 ? iC : 0.0;
  qn_pams.penalty_normalized  = true;
  qn_pams.max_iter            = params.max_iter;
  qn_pams.grad_tol            = params.grad_tol;
  qn_pams.change_tol          = params.change_tol;
  qn_pams.linesearch_max_iter = params.linesearch_max_iter;
  qn_pams.lbfgs_memory        = params.lbfgs_memory;
  qn_pams.verbose             = static_cast<int>(params.verbose);

  T* y1 = (T*)y;
  T* w1 = w;
  rmm::device_uvector<T> y1Buf(0, stream);
  rmm::device_uvector<T> w1Buf(0, stream);
  if (nClasses > 1) {
    y1Buf.resize(nRows * coefCols, stream);
    y1 = y1Buf.data();
  }
  if (coefCols > 1) {
    w1Buf.resize(coefCols * coefRows, stream);
    w1 = w1Buf.data();
  }
  RAFT_CUDA_TRY(cudaMemsetAsync(w1, 0, coefCols * coefRows * sizeof(T), stream));

  // one-vs-rest logic goes over each class
  const int n_streams = coefCols > 1 ? handle.get_stream_pool_size() : 1;
  bool parallel       = n_streams > 1;
  int max_n_iter      = 0;
#pragma omp parallel for num_threads(n_streams) if (parallel) reduction(max : max_n_iter)
  for (int class_i = 0; class_i < coefCols; class_i++) {
    T* yi = y1 + nRows * class_i;
    T* wi = w1 + coefRows * class_i;
    auto worker =
      parallel ? WorkerHandle(handle, omp_get_thread_num()) : WorkerHandle(handle, stream);
    if (nClasses > 1) {
      raft::linalg::unaryOp(
        yi, y, nRows, OvrSelector<T>{classes, nClasses == 2 ? 1 : class_i}, worker.stream);
    }
    T target;
    int n_iter;
    GLM::qnFit<T>(worker.handle,
                  qn_pams,
                  X1,
                  true,
                  yi,
                  narrowDown(nRows),
                  narrowDown(nCols1),
                  // regression: C == 1; classification: C == 2
                  nClasses == 0 ? 1 : 2,
                  wi,
                  &target,
                  &n_iter,
                  (T*)sampleWeight,
                  T(params.epsilon));
    if (n_iter > max_n_iter) { max_n_iter = n_iter; }
  }
  if (parallel) handle.sync_stream_pool();

  if (coefCols > 1) { raft::linalg::transpose(handle, w1, w, coefRows, coefCols, stream); }

  return max_n_iter;
}

// Explicit instantiations for library
template CUML_EXPORT int fit<float>(const raft::handle_t& handle,
                                    const Params& params,
                                    const std::size_t nRows,
                                    const std::size_t nCols,
                                    const int nClasses,
                                    const float* classes,
                                    const float* X,
                                    const float* y,
                                    const float* sampleWeight,
                                    float* w);
template CUML_EXPORT int fit<double>(const raft::handle_t& handle,
                                     const Params& params,
                                     const std::size_t nRows,
                                     const std::size_t nCols,
                                     const int nClasses,
                                     const double* classes,
                                     const double* X,
                                     const double* y,
                                     const double* sampleWeight,
                                     double* w);
}  // namespace linear
}  // namespace SVM
}  // namespace ML
