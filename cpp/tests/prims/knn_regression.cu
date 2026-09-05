/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.h"

#include <raft/label/classlabels.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cuvs/neighbors/brute_force.hpp>
#include <gtest/gtest.h>
#include <selection/knn.cuh>

#include <iostream>
#include <vector>

namespace MLCommon {
namespace Selection {

struct KNNRegressionInputs {
  int rows;
  int cols;
  int n_labels;
  float cluster_std;
  int k;
};

void generate_data(
  float* out_samples, float* out_labels, int n_rows, int n_cols, cudaStream_t stream)
{
  raft::random::Rng r(0ULL, raft::random::GenPC);

  r.uniform(out_samples, n_rows * n_cols, 0.0f, 1.0f, stream);

  raft::linalg::unaryOp<float>(
    out_samples,
    out_samples,
    n_rows,
    [=] __device__(float input) { return 2 * input - 1; },
    stream);

  raft::linalg::reduce<true, true>(
    out_labels,
    out_samples,
    n_cols,
    n_rows,
    0.0f,
    stream,
    false,
    [=] __device__(float in, int n) { return in * in; },
    raft::add_op(),
    [=] __device__(float in) { return sqrt(in); });

  thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(out_labels);
  float max = *(thrust::max_element(thrust::cuda::par.on(stream), d_ptr, d_ptr + n_rows));

  raft::linalg::unaryOp<float>(
    out_labels, out_labels, n_rows, [=] __device__(float input) { return input / max; }, stream);
}

class KNNRegressionTest : public ::testing::TestWithParam<KNNRegressionInputs> {
 public:
  KNNRegressionTest()
    : params(::testing::TestWithParam<KNNRegressionInputs>::GetParam()),
      stream(handle.get_stream().get()),
      train_samples(params.rows * params.cols, stream),
      train_labels(params.rows, stream),
      pred_labels(params.rows, stream),
      knn_indices(params.rows * params.k, stream),
      knn_dists(params.rows * params.k, stream)
  {
  }

 protected:
  void basicTest()
  {
    generate_data(train_samples.data(), train_labels.data(), params.rows, params.cols, stream);

    auto train_view = raft::make_device_matrix_view<const float, int64_t>(
      train_samples.data(), params.rows, params.cols);

    auto idx = cuvs::neighbors::brute_force::build(
      handle, train_view, cuvs::distance::DistanceType::L2Unexpanded);

    cuvs::neighbors::brute_force::search(
      handle,
      idx,
      train_view,
      raft::make_device_matrix_view<int64_t, int64_t>(knn_indices.data(), params.rows, params.k),
      raft::make_device_matrix_view<float, int64_t>(knn_dists.data(), params.rows, params.k));

    std::vector<float*> y;
    y.push_back(train_labels.data());

    knn_regress(
      handle, pred_labels.data(), knn_indices.data(), y, params.rows, params.rows, params.k);

    handle.sync_stream(stream);
  }

  void SetUp() override { basicTest(); }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  KNNRegressionInputs params;

  rmm::device_uvector<float> train_samples;
  rmm::device_uvector<float> train_labels;

  rmm::device_uvector<float> pred_labels;

  rmm::device_uvector<int64_t> knn_indices;
  rmm::device_uvector<float> knn_dists;
};

typedef KNNRegressionTest KNNRegressionTestF;
TEST_P(KNNRegressionTestF, Fit)
{
  ASSERT_TRUE(devArrMatch(
    train_labels.data(), pred_labels.data(), params.rows, MLCommon::CompareApprox<float>(0.3)));
}

const std::vector<KNNRegressionInputs> inputsf = {{100, 10, 2, 0.01f, 2},
                                                  {1000, 10, 5, 0.01f, 2},
                                                  {10000, 10, 5, 0.01f, 2},
                                                  {100, 10, 2, 0.01f, 10},
                                                  {1000, 10, 5, 0.01f, 10},
                                                  {10000, 10, 5, 0.01f, 10},
                                                  {100, 10, 2, 0.01f, 15},
                                                  {1000, 10, 5, 0.01f, 15},
                                                  {10000, 10, 5, 0.01f, 15}};

INSTANTIATE_TEST_CASE_P(KNNRegressionTest, KNNRegressionTestF, ::testing::ValuesIn(inputsf));

};  // end namespace Selection
};  // namespace MLCommon
