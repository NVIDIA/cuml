/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.h"

#include <raft/label/classlabels.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuvs/neighbors/brute_force.hpp>
#include <gtest/gtest.h>
#include <selection/knn.cuh>

#include <iostream>
#include <vector>

namespace MLCommon {
namespace Selection {

struct KNNClassifyInputs {
  int rows;
  int cols;
  int n_labels;
  float cluster_std;
  int k;
};

class KNNClassifyTest : public ::testing::TestWithParam<KNNClassifyInputs> {
 public:
  KNNClassifyTest()
    : params(::testing::TestWithParam<KNNClassifyInputs>::GetParam()),
      stream(handle.get_stream().get()),
      train_samples(params.rows * params.cols, stream),
      train_labels(params.rows, stream),
      pred_labels(params.rows, stream),
      knn_indices(params.rows * params.k, stream),
      knn_dists(params.rows * params.k, stream)
  {
    basicTest();
  }

 protected:
  void basicTest()
  {
    raft::random::make_blobs<float, int>(train_samples.data(),
                                         train_labels.data(),
                                         params.rows,
                                         params.cols,
                                         params.n_labels,
                                         stream,
                                         true,
                                         nullptr,
                                         nullptr,
                                         params.cluster_std);

    rmm::device_uvector<int> unique_labels(0, stream);
    auto n_classes =
      raft::label::getUniquelabels(unique_labels, train_labels.data(), params.rows, stream);

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

    std::vector<int*> y;
    y.push_back(train_labels.data());

    std::vector<int*> uniq_labels;
    uniq_labels.push_back(unique_labels.data());

    std::vector<int> n_unique;
    n_unique.push_back(n_classes);

    knn_classify(handle,
                 pred_labels.data(),
                 knn_indices.data(),
                 y,
                 params.rows,
                 params.rows,
                 params.k,
                 uniq_labels,
                 n_unique);

    handle.sync_stream(stream);
  }

 protected:
  KNNClassifyInputs params;
  raft::handle_t handle;
  cudaStream_t stream;

  rmm::device_uvector<float> train_samples;
  rmm::device_uvector<int> train_labels;

  rmm::device_uvector<int> pred_labels;

  rmm::device_uvector<int64_t> knn_indices;
  rmm::device_uvector<float> knn_dists;
};

typedef KNNClassifyTest KNNClassifyTestF;
TEST_P(KNNClassifyTestF, Fit)
{
  ASSERT_TRUE(
    devArrMatch(train_labels.data(), pred_labels.data(), params.rows, MLCommon::Compare<int>()));
}

const std::vector<KNNClassifyInputs> inputsf = {{100, 10, 2, 0.01f, 2},
                                                {1000, 10, 5, 0.01f, 2},
                                                {10000, 10, 5, 0.01f, 2},
                                                {100, 10, 2, 0.01f, 10},
                                                {1000, 10, 5, 0.01f, 10},
                                                {10000, 10, 5, 0.01f, 10},
                                                {100, 10, 2, 0.01f, 50},
                                                {1000, 10, 5, 0.01f, 50},
                                                {10000, 10, 5, 0.01f, 50}};

INSTANTIATE_TEST_CASE_P(KNNClassifyTest, KNNClassifyTestF, ::testing::ValuesIn(inputsf));

};  // end namespace Selection
};  // namespace MLCommon
