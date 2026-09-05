/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/util/cudart_utils.hpp>

#include <glm/qn/glm_base.cuh>
#include <glm/qn/glm_logistic.cuh>
#include <glm/qn/glm_regularizer.cuh>
#include <glm/qn/mg/standardization.cuh>
#include <glm/qn/qn_solvers.cuh>
#include <glm/qn/qn_util.cuh>

#include <iostream>
#include <vector>

namespace ML {
namespace GLM {
namespace opg {
template <typename T>
// multi-gpu version of linearBwd
inline void linearBwdMG(const raft::handle_t& handle,
                        SimpleDenseMat<T>& G,
                        const SimpleMat<T>& X,
                        const SimpleDenseMat<T>& dZ,
                        bool setZero,
                        const int64_t n_samples,
                        const int n_ranks)
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
    Gweights.assign_gemm(handle, 1.0 / n_samples, dZ, false, X, false, beta / n_ranks, stream);

    raft::stats::mean<true>(Gbias.data, dZ.data, dZ.m, dZ.n, false, stream);
    T bias_factor = 1.0 * dZ.n / n_samples;
    raft::linalg::multiplyScalar(Gbias.data, Gbias.data, bias_factor, dZ.m, stream);

  } else {
    CUML_LOG_DEBUG("has bias not enabled");
    G.assign_gemm(handle, 1.0 / n_samples, dZ, false, X, false, beta / n_ranks, stream);
  }
}

/**
 * @brief Aggregates local gradient vectors and loss values from local training data. This
 * class is the multi-node-multi-gpu version of GLMWithData.
 *
 * The implementation overrides existing GLMWithData::() function. The purpose is to
 * aggregate local gradient vectors and loss values from distributed X, y, where X represents the
 * input vectors and y represents labels.
 *
 * GLMWithData::() currently invokes three functions: linearFwd, getLossAndDz and linearBwd.
 * linearFwd multiplies local input vectors with the coefficient vector (i.e. coef_), so does not
 * require communication. getLossAndDz calculates local loss so requires allreduce to obtain a
 * global loss. linearBwd calculates local gradient vector so requires allreduce to obtain a
 * global gradient vector. The global loss and the global gradient vector will be used in
 * min_lbfgs to update coefficient. The update runs individually on every GPU and when finished,
 * all GPUs have the same value of coefficient.
 */
template <typename T, class GLMObjective>
struct GLMWithDataMG : ML::GLM::detail::GLMWithData<T, GLMObjective> {
  const raft::handle_t* handle_p;
  int rank;
  int64_t n_samples;
  int n_ranks;
  const Standardizer<T>* stder_p;

  GLMWithDataMG(raft::handle_t const& handle,
                int rank,
                int n_ranks,
                int64_t n_samples,
                GLMObjective* obj,
                const SimpleMat<T>& X,
                const SimpleVec<T>& y,
                SimpleDenseMat<T>& Z,
                const Standardizer<T>* stder_p = NULL)
    : ML::GLM::detail::GLMWithData<T, GLMObjective>(obj, X, y, Z)
  {
    this->handle_p  = &handle;
    this->rank      = rank;
    this->n_ranks   = n_ranks;
    this->n_samples = n_samples;
    this->stder_p   = stder_p;
  }

  inline T operator()(const SimpleVec<T>& wFlat,
                      SimpleVec<T>& gradFlat,
                      T* dev_scalar,
                      cudaStream_t stream)
  {
    raft::comms::comms_t const& communicator = raft::resource::get_comms(*(this->handle_p));
    SimpleDenseMat<T> W(wFlat.data, this->C, this->dims);
    SimpleDenseMat<T> G(gradFlat.data, this->C, this->dims);
    SimpleVec<T> lossVal(dev_scalar, 1);

    // Ensure the same coefficients on all GPU
    communicator.bcast(wFlat.data, this->C * this->dims, 0, stream);
    communicator.sync_stream(stream);

    // apply regularization
    auto regularizer_obj = this->objective;
    auto lossFunc        = regularizer_obj->loss;
    auto reg             = regularizer_obj->reg;
    G.fill(0, stream);
    T reg_host = 0;
    if (reg->l2_penalty != 0) {
      reg->reg_grad(dev_scalar, G, W, lossFunc->fit_intercept, stream);
      raft::update_host(&reg_host, dev_scalar, 1, stream);
      raft::resource::sync_stream(*(this->handle_p));
    }

    // if standardization is True
    std::vector<T> wFlatOrigin(this->C * this->dims);
    if (stder_p != NULL) {
      raft::copy(wFlatOrigin.data(), wFlat.data, this->C * this->dims, stream);

      stder_p->adapt_model_for_linearFwd(
        *handle_p, wFlat.data, this->C, (this->X)->n, (this->X)->n != G.n);

      // scale reg part of the gradient for the upcoming adapt_gradient_for_linearBwd
      raft::linalg::matrixVectorOp<false, true>(
        G.data, G.data, stder_p->std.data, stder_p->std.len, G.m, raft::mul_op(), stream);
    }

    // apply linearFwd, getLossAndDz, linearBwd
    ML::GLM::detail::linearFwd(
      lossFunc->handle, *(this->Z), *(this->X), W);  // linear part: forward pass

    lossFunc->getLossAndDZ(dev_scalar, *(this->Z), *(this->y), stream);  // loss specific part

    // normalize local loss before allreduce sum
    T factor = 1.0 * (*this->y).len / this->n_samples;
    raft::linalg::multiplyScalar(dev_scalar, dev_scalar, factor, 1, stream);

    // GPUs calculates reg_host independently and may get values that show tiny divergence.
    // Take the averaged reg_host to avoid the divergence.
    T reg_factor = reg_host / this->n_ranks;
    raft::linalg::addScalar(dev_scalar, dev_scalar, reg_factor, 1, stream);

    communicator.allreduce(dev_scalar, dev_scalar, 1, raft::comms::op_t::SUM, stream);
    communicator.sync_stream(stream);

    linearBwdMG(lossFunc->handle,
                G,
                *(this->X),
                *(this->Z),
                false,
                n_samples,
                n_ranks);  // linear part: backward pass

    communicator.allreduce(G.data, G.data, this->C * this->dims, raft::comms::op_t::SUM, stream);
    communicator.sync_stream(stream);

    if (stder_p != NULL) {
      stder_p->adapt_gradient_for_linearBwd(*handle_p, G, *(this->Z), (this->X)->n != G.n);
      raft::copy(wFlat.data, wFlatOrigin.data(), this->C * this->dims, stream);
    }

    T loss_host;
    raft::update_host(&loss_host, dev_scalar, 1, stream);
    raft::resource::sync_stream(*(this->handle_p));

    return loss_host;
  }
};
};  // namespace opg
};  // namespace GLM
};  // namespace ML
