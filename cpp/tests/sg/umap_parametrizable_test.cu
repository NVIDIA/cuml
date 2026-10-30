/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/distance_type.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/manifold/umap.hpp>
#include <cuml/manifold/umapparams.h>
#include <cuml/metrics/metrics.hpp>
#include <cuml/neighbors/knn.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_coo_matrix.hpp>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_buffer.hpp>

#include <cuvs/distance/distance.hpp>
#include <datasets/digits.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include <umap/runner.cuh>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

using namespace ML;
using namespace ML::Metrics;

using namespace MLCommon;
using namespace MLCommon::Datasets::Digits;

template <typename T>
CUML_KERNEL void has_nan_kernel(T* data, size_t len, bool* answer)
{
  static_assert(std::is_floating_point<T>());
  std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if ((tid < len) && isnan(data[tid])) { *answer = true; }
}

template <typename T>
bool has_nan(T* data, size_t len, cudaStream_t stream)
{
  dim3 blk(256);
  dim3 grid(raft::ceildiv(len, (size_t)blk.x));
  bool h_answer = false;
  rmm::device_scalar<bool> d_answer(stream);
  raft::update_device(d_answer.data(), &h_answer, 1, stream);
  has_nan_kernel<<<grid, blk, 0, stream>>>(data, len, d_answer.data());
  h_answer = d_answer.value(stream);
  return h_answer;
}

template <typename T>
CUML_KERNEL void are_equal_kernel(T* embedding1, T* embedding2, size_t len, double* diff)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (embedding1[tid] != embedding2[tid]) {
    atomicAdd(diff, abs(embedding1[tid] - embedding2[tid]));
  }
}

template <typename T>
bool are_equal(T* embedding1, T* embedding2, size_t len, cudaStream_t stream)
{
  double h_answer = 0.;
  rmm::device_scalar<double> d_answer(stream);
  raft::update_device(d_answer.data(), &h_answer, 1, stream);
  are_equal_kernel<<<raft::ceildiv(len, (size_t)32), 32, 0, stream>>>(
    embedding1, embedding2, len, d_answer.data());
  h_answer = d_answer.value(stream);

  double tolerance = 1.0;
  if (h_answer > tolerance) {
    std::cout << "Not equal, difference : " << h_answer << std::endl;
    return false;
  }
  return true;
}

class UMAPParametrizableTest : public ::testing::Test {
 protected:
  struct TestParams {
    bool fit_transform;
    bool supervised;
    bool knn_params;
    bool refine;
    int n_samples;
    int n_features;
    int n_clusters;
    double min_trustworthiness;
  };

  void get_embedding(raft::handle_t& handle,
                     float* X,
                     float* y,
                     float* embedding_ptr,
                     TestParams& test_params,
                     UMAPParams& umap_params)
  {
    cudaStream_t stream = handle.get_stream().get();
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    rmm::device_uvector<int64_t>* knn_indices_b{};
    rmm::device_uvector<float>* knn_dists_b{};
    int64_t* knn_indices{};
    float* knn_dists{};
    if (test_params.knn_params) {
      knn_indices_b = new rmm::device_uvector<int64_t>(n_samples * umap_params.n_neighbors, stream);
      knn_dists_b   = new rmm::device_uvector<float>(n_samples * umap_params.n_neighbors, stream);
      knn_indices   = knn_indices_b->data();
      knn_dists     = knn_dists_b->data();

      auto X_view = raft::make_device_matrix_view<const float, int64_t>(X, n_samples, n_features);
      auto idx    = cuvs::neighbors::brute_force::build(
        handle, X_view, cuvs::distance::DistanceType::L2Unexpanded);

      cuvs::neighbors::brute_force::search(handle,
                                           idx,
                                           X_view,
                                           raft::make_device_matrix_view<int64_t, int64_t>(
                                             knn_indices, n_samples, umap_params.n_neighbors),
                                           raft::make_device_matrix_view<float, int64_t>(
                                             knn_dists, n_samples, umap_params.n_neighbors));

      handle.sync_stream(stream);
    }

    std::unique_ptr<rmm::device_buffer> model_embedding_buffer;

    auto graph =
      raft::make_host_coo_matrix<float, int, int, uint64_t>(handle, n_samples, n_samples);

    if (test_params.supervised) {
      ML::UMAP::fit(handle,
                    X,
                    y,
                    n_samples,
                    n_features,
                    knn_indices,
                    knn_dists,
                    &umap_params,
                    model_embedding_buffer,
                    graph);
    } else {
      ML::UMAP::fit(handle,
                    X,
                    nullptr,
                    n_samples,
                    n_features,
                    knn_indices,
                    knn_dists,
                    &umap_params,
                    model_embedding_buffer,
                    graph);
    }

    // Extract pointer from device_buffer after fit allocates and fills it
    float* model_embedding = static_cast<float*>(model_embedding_buffer->data());

    if (test_params.refine) {
      std::cout << "using refine";
      if (test_params.supervised) {
        auto cgraph_coo =
          ML::UMAP::get_graph(handle, X, y, n_samples, n_features, nullptr, nullptr, &umap_params);
        ML::UMAP::refine(
          handle, X, n_samples, n_features, cgraph_coo.get(), &umap_params, model_embedding);
      } else {
        auto cgraph_coo = ML::UMAP::get_graph(
          handle, X, nullptr, n_samples, n_features, nullptr, nullptr, &umap_params);
        ML::UMAP::refine(
          handle, X, n_samples, n_features, cgraph_coo.get(), &umap_params, model_embedding);
      }
    }
    handle.sync_stream(stream);

    if (test_params.fit_transform) {
      // Copy the model embedding to the output embedding_ptr
      raft::copy(embedding_ptr, model_embedding, n_samples * umap_params.n_components, stream);
      handle.sync_stream(stream);
    } else {
      // Use transform for non-fit_transform case
      RAFT_CUDA_TRY(cudaMemsetAsync(
        embedding_ptr, 0, n_samples * umap_params.n_components * sizeof(float), stream));

      handle.sync_stream(stream);

      ML::UMAP::transform(handle,
                          X,
                          n_samples,
                          umap_params.n_components,
                          X,
                          n_samples,
                          model_embedding,
                          n_samples,
                          &umap_params,
                          embedding_ptr);

      handle.sync_stream(stream);
    }

    if (test_params.knn_params) {
      delete knn_indices_b;
      delete knn_dists_b;
    }
  }

  void assertions(raft::handle_t& handle,
                  float* X,
                  float* embedding_ptr,
                  TestParams& test_params,
                  UMAPParams& umap_params)
  {
    cudaStream_t stream = handle.get_stream().get();
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    ASSERT_TRUE(!has_nan(embedding_ptr, n_samples * umap_params.n_components, stream));

    double trustworthiness =
      trustworthiness_score<float, ML::distance::DistanceType::L2SqrtUnexpanded>(
        handle,
        X,
        embedding_ptr,
        n_samples,
        n_features,
        umap_params.n_components,
        umap_params.n_neighbors);

    std::cout << "min. expected trustworthiness: " << test_params.min_trustworthiness << std::endl;
    std::cout << "trustworthiness: " << trustworthiness << std::endl;
    ASSERT_TRUE(trustworthiness > test_params.min_trustworthiness);
  }

  void test(TestParams& test_params, UMAPParams& umap_params)
  {
    std::cout << "\numap_params : [" << std::boolalpha << umap_params.n_neighbors << "-"
              << umap_params.n_components << "-" << umap_params.n_epochs << "-"
              << umap_params.random_state << std::endl;

    std::cout << "test_params : [" << std::boolalpha << test_params.fit_transform << "-"
              << test_params.supervised << "-" << test_params.refine << "-"
              << test_params.knn_params << "-" << test_params.n_samples << "-"
              << test_params.n_features << "-" << test_params.n_clusters << "-"
              << test_params.min_trustworthiness << "]" << std::endl;

    raft::handle_t handle;
    cudaStream_t stream = handle.get_stream().get();
    int& n_samples      = test_params.n_samples;
    int& n_features     = test_params.n_features;

    UMAP::find_ab(handle, &umap_params);

    rmm::device_uvector<float> X_d(n_samples * n_features, stream);
    rmm::device_uvector<int> y_d(n_samples, stream);

    ML::Datasets::make_blobs(handle,
                             X_d.data(),
                             y_d.data(),
                             n_samples,
                             n_features,
                             test_params.n_clusters,
                             true,
                             nullptr,
                             nullptr,
                             1.f,
                             true,
                             -10.f,
                             10.f,
                             1234ULL);

    handle.sync_stream(stream);

    raft::linalg::convert_array((float*)y_d.data(), y_d.data(), n_samples, stream);

    handle.sync_stream(stream);

    rmm::device_uvector<float> embeddings1(n_samples * umap_params.n_components, stream);

    float* e1 = embeddings1.data();

    umap_params.init         = 0;
    umap_params.random_state = 43;
    umap_params.n_epochs     = 500;
    get_embedding(handle, X_d.data(), (float*)y_d.data(), e1, test_params, umap_params);

    assertions(handle, X_d.data(), e1, test_params, umap_params);

    // Disable reproducibility tests after transformation
    if (!test_params.fit_transform) { return; }

    rmm::device_uvector<float> embeddings2(n_samples * umap_params.n_components, stream);
    float* e2 = embeddings2.data();
    get_embedding(handle, X_d.data(), (float*)y_d.data(), e2, test_params, umap_params);

    auto equal = are_equal(e1, e2, n_samples * umap_params.n_components, stream);

    if (!equal) {
      raft::print_device_vector("e1", e1, 25, std::cout);
      raft::print_device_vector("e2", e2, 25, std::cout);
    }

    ASSERT_TRUE(equal);
  }

  void SetUp() override
  {
    std::vector<TestParams> test_params_vec = {{false, false, false, true, 2000, 50, 20, 0.45},
                                               {true, false, false, false, 2000, 50, 20, 0.45},
                                               {false, true, false, true, 2000, 50, 20, 0.45},
                                               {false, false, true, false, 2000, 50, 20, 0.45},
                                               {true, true, false, true, 2000, 50, 20, 0.45},
                                               {true, false, true, false, 2000, 50, 20, 0.45},
                                               {false, true, true, true, 2000, 50, 20, 0.45},
                                               {true, true, true, false, 2000, 50, 20, 0.45}};

    std::vector<UMAPParams> umap_params_vec(4);
    umap_params_vec[0].n_components = 2;

    umap_params_vec[1].n_components = 10;

    umap_params_vec[2].n_components = 21;
    umap_params_vec[2].random_state = 43;
    umap_params_vec[2].init         = 0;
    umap_params_vec[2].n_epochs     = 500;

    umap_params_vec[3].n_components = 25;
    umap_params_vec[3].random_state = 43;
    umap_params_vec[3].init         = 0;
    umap_params_vec[3].n_epochs     = 500;

    for (auto& umap_params : umap_params_vec) {
      for (auto& test_params : test_params_vec) {
        test(test_params, umap_params);
      }
    }
  }

  void TearDown() override {}
};

typedef UMAPParametrizableTest UMAPParametrizableTest;
TEST_F(UMAPParametrizableTest, Result) {}

namespace {

struct SmoothKnnResult {
  std::vector<float> sigmas;
  std::vector<float> rhos;
};

template <typename Index>
SmoothKnnResult smooth_knn_distances(const std::vector<Index>& indices,
                                     const std::vector<float>& distances,
                                     int n_neighbors,
                                     float local_connectivity,
                                     bool self_knn = true)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();
  int n       = static_cast<int>(distances.size() / n_neighbors);
  rmm::device_uvector<Index> indices_d(indices.size(), stream);
  rmm::device_uvector<float> distances_d(distances.size(), stream);
  rmm::device_uvector<float> sigmas_d(n, stream);
  rmm::device_uvector<float> rhos_d(n, stream);
  raft::update_device(indices_d.data(), indices.data(), indices.size(), stream);
  raft::update_device(distances_d.data(), distances.data(), distances.size(), stream);

  UMAPParams params;
  UMAPAlgo::FuzzySimplSet::Naive::smooth_knn_dist<float, Index, int64_t, 128>(n,
                                                                              indices_d.data(),
                                                                              distances_d.data(),
                                                                              rhos_d.data(),
                                                                              sigmas_d.data(),
                                                                              &params,
                                                                              n_neighbors,
                                                                              local_connectivity,
                                                                              stream.get(),
                                                                              self_knn);

  SmoothKnnResult result{std::vector<float>(n), std::vector<float>(n)};
  std::vector<Index> indices_after(indices.size());
  std::vector<float> distances_after(distances.size());
  raft::update_host(result.sigmas.data(), sigmas_d.data(), n, stream);
  raft::update_host(result.rhos.data(), rhos_d.data(), n, stream);
  raft::update_host(indices_after.data(), indices_d.data(), indices.size(), stream);
  raft::update_host(distances_after.data(), distances_d.data(), distances.size(), stream);
  handle.sync_stream(stream);
  EXPECT_EQ(indices_after, indices);
  EXPECT_EQ(distances_after, distances);
  return result;
}

template <typename Index>
class UMAPSmoothKnnTest : public ::testing::Test {};

using UMAPSmoothKnnIndexTypes = ::testing::Types<int, int64_t>;
TYPED_TEST_SUITE(UMAPSmoothKnnTest, UMAPSmoothKnnIndexTypes);

TYPED_TEST(UMAPSmoothKnnTest, RoundedSelfDistances)
{
  constexpr int n                      = 8;
  constexpr int n_neighbors            = 4;
  constexpr float neighbor_distances[] = {0.0f, 1e-4f, 0.2f, 0.4f};
  std::vector<TypeParam> indices(n * n_neighbors);
  std::vector<float> distances(n * n_neighbors);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n_neighbors; ++col) {
      indices[row * n_neighbors + col]   = (row + col) % n;
      distances[row * n_neighbors + col] = neighbor_distances[col];
    }
  }
  // This row takes its bandwidth floor from the global mean.
  std::fill_n(distances.begin(), n_neighbors, 0.0f);

  for (float local_connectivity : {0.0f, 0.5f, 1.0f, 1.5f, 3.5f}) {
    SCOPED_TRACE(local_connectivity);
    auto expected      = smooth_knn_distances(indices, distances, n_neighbors, local_connectivity);
    float expected_rho = local_connectivity <= 1.0f   ? local_connectivity * 1e-4f
                         : local_connectivity == 1.5f ? 1e-4f + 0.5f * (0.2f - 1e-4f)
                                                      : 0.4f;
    EXPECT_FLOAT_EQ(expected.rhos[1], expected_rho);
    EXPECT_NEAR(
      expected.sigmas[0], 1e-3f * 7.0f * (1e-4f + 0.2f + 0.4f) / (n * n_neighbors), 1e-9f);

    for (int self_column : {0, 1, 3}) {
      SCOPED_TRACE(self_column);
      auto rounded_indices   = indices;
      auto rounded_distances = distances;
      for (int row = 0; row < n; ++row) {
        int start                = row * n_neighbors;
        rounded_distances[start] = self_column == 0 ? 1e-5f : self_column == 1 ? 0.02f : 0.5f;
        // Keep the supplied distances sorted, including the all-zero row.
        int column = row == 0 ? n_neighbors - 1 : self_column;
        std::rotate(rounded_indices.begin() + start,
                    rounded_indices.begin() + start + 1,
                    rounded_indices.begin() + start + column + 1);
        std::rotate(rounded_distances.begin() + start,
                    rounded_distances.begin() + start + 1,
                    rounded_distances.begin() + start + column + 1);
      }
      auto actual =
        smooth_knn_distances(rounded_indices, rounded_distances, n_neighbors, local_connectivity);
      for (int row = 0; row < n; ++row) {
        EXPECT_FLOAT_EQ(actual.rhos[row], expected.rhos[row]);
        EXPECT_NEAR(actual.sigmas[row], expected.sigmas[row], 1e-6f * expected.sigmas[row] + 1e-9f);
      }
    }
  }
}

TYPED_TEST(UMAPSmoothKnnTest, TransformDistances)
{
  constexpr int n                      = 8;
  constexpr int n_neighbors            = 4;
  constexpr float neighbor_distances[] = {0.01f, 0.02f, 0.2f, 0.4f};
  std::vector<TypeParam> indices(n * n_neighbors);
  std::vector<float> distances(n * n_neighbors);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n_neighbors; ++col) {
      // The neighbor ID matches the query row number in column 1. This is not
      // a self entry: queries and neighbors have separate index spaces.
      indices[row * n_neighbors + col]   = (row + col + n - 1) % n;
      distances[row * n_neighbors + col] = neighbor_distances[col];
    }
  }
  std::fill_n(distances.begin(), n_neighbors, 0.0f);
  auto actual = smooth_knn_distances(indices, distances, n_neighbors, 1.0f, false);
  // An index array without row/neighbor coincidences must give the same result.
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n_neighbors; ++col) {
      indices[row * n_neighbors + col] = (row + col + 1) % n;
    }
  }
  auto expected = smooth_knn_distances(indices, distances, n_neighbors, 1.0f, false);
  for (int row = 0; row < n; ++row) {
    EXPECT_FLOAT_EQ(actual.rhos[row], row == 0 ? 0.0f : 0.01f);
    EXPECT_FLOAT_EQ(actual.rhos[row], expected.rhos[row]);
    EXPECT_FLOAT_EQ(actual.sigmas[row], expected.sigmas[row]);
  }
}

}  // namespace
