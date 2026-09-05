/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/utils.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/neighbors/knn.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <iostream>
#include <vector>

namespace ML {

using namespace raft::random;
using namespace std;

struct KNNInputs {
  int n_rows;
  int n_cols;
  int n_centers;

  int n_query_row;

  int n_neighbors;
  int n_parts;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const KNNInputs& dims)
{
  return os;
}

template <typename T>
void gen_blobs(
  raft::handle_t& handle, T* out, int* l, int rows, int cols, int centers, const T* centroids)
{
  Datasets::make_blobs(handle,
                       out,
                       l,
                       rows,
                       cols,
                       centers,
                       true,
                       centroids,
                       nullptr,
                       0.1f,
                       true,
                       -10.0f,
                       10.0f,
                       1234ULL);
}

void create_index_parts(raft::handle_t& handle,
                        float* query_data,
                        int* query_labels,
                        vector<float*>& part_inputs,
                        vector<int*>& part_labels,
                        vector<int>& part_sizes,
                        const KNNInputs& params,
                        const float* centers)
{
  cudaStream_t stream = handle.get_stream().get();
  gen_blobs<float>(handle,
                   query_data,
                   query_labels,
                   params.n_rows * params.n_parts,
                   params.n_cols,
                   params.n_centers,
                   centers);

  for (int i = 0; i < params.n_parts; i++) {
    part_inputs.push_back(query_data + (i * params.n_rows * params.n_cols));
    part_labels.push_back(query_labels + (i * params.n_rows));
    part_sizes.push_back(params.n_rows);
  }
}

CUML_KERNEL void to_float(float* out, int* in, int size)
{
  int element = threadIdx.x + blockDim.x * blockIdx.x;
  if (element >= size) return;
  out[element] = float(in[element]);
}

CUML_KERNEL void build_actual_output(
  int* output, int n_rows, int k, const int* idx_labels, const int64_t* indices)
{
  int element = threadIdx.x + blockDim.x * blockIdx.x;
  if (element >= n_rows * k) return;

  int ind         = (int)indices[element];
  output[element] = idx_labels[ind];
}

CUML_KERNEL void build_expected_output(int* output, int n_rows, int k, const int* labels)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= n_rows) return;

  int cur_label = labels[row];
  for (int i = 0; i < k; i++) {
    output[row * k + i] = cur_label;
  }
}

template <typename T>
class KNNTest : public ::testing::TestWithParam<KNNInputs> {
 public:
  KNNTest()
    : params(::testing::TestWithParam<KNNInputs>::GetParam()),
      stream(handle.get_stream().get()),
      index_data(params.n_rows * params.n_cols * params.n_parts, stream),
      index_labels(params.n_rows * params.n_parts, stream),
      search_data(params.n_query_row * params.n_cols, stream),
      search_labels(params.n_query_row, stream),
      output_indices(params.n_query_row * params.n_neighbors * params.n_parts, stream),
      output_dists(params.n_query_row * params.n_neighbors * params.n_parts, stream)

  {
    RAFT_CUDA_TRY(cudaMemsetAsync(index_data.data(), 0, index_data.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(index_labels.data(), 0, index_labels.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(search_data.data(), 0, search_data.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(search_labels.data(), 0, search_labels.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(output_indices.data(), 0, output_indices.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(output_dists.data(), 0, output_dists.size() * sizeof(T), stream));
  }

 protected:
  void testBruteForce()
  {
    rmm::device_uvector<int> actual_labels(params.n_query_row * params.n_neighbors * params.n_parts,
                                           stream);
    rmm::device_uvector<int> expected_labels(
      params.n_query_row * params.n_neighbors * params.n_parts, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(actual_labels.data(), 0, actual_labels.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(expected_labels.data(), 0, expected_labels.size() * sizeof(T), stream));

    create_data();

    brute_force_knn(handle,
                    part_inputs,
                    part_sizes,
                    params.n_cols,
                    search_data.data(),
                    params.n_query_row,
                    output_indices.data(),
                    output_dists.data(),
                    params.n_neighbors,
                    true,
                    true);

    build_actual_output<<<raft::ceildiv(params.n_query_row * params.n_neighbors, 32),
                          32,
                          0,
                          stream>>>(actual_labels.data(),
                                    params.n_query_row,
                                    params.n_neighbors,
                                    index_labels.data(),
                                    output_indices.data());

    build_expected_output<<<raft::ceildiv(params.n_query_row, 32), 32, 0, stream>>>(
      expected_labels.data(), params.n_query_row, params.n_neighbors, search_labels.data());

    ASSERT_TRUE(devArrMatch(expected_labels.data(),
                            actual_labels.data(),
                            params.n_query_row * params.n_neighbors,
                            MLCommon::Compare<int>()));
  }

  void testClassification()
  {
    rmm::device_uvector<int> actual_labels(params.n_query_row, stream);
    rmm::device_uvector<int> expected_labels(params.n_query_row, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(actual_labels.data(), 0, actual_labels.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(expected_labels.data(), 0, expected_labels.size() * sizeof(T), stream));

    create_data();

    brute_force_knn(handle,
                    part_inputs,
                    part_sizes,
                    params.n_cols,
                    search_data.data(),
                    params.n_query_row,
                    output_indices.data(),
                    output_dists.data(),
                    params.n_neighbors,
                    true,
                    true);

    vector<int*> full_labels(1);
    full_labels[0] = index_labels.data();

    knn_classify(handle,
                 actual_labels.data(),
                 output_indices.data(),
                 full_labels,
                 params.n_rows * params.n_parts,
                 params.n_query_row,
                 params.n_neighbors);

    ASSERT_TRUE(devArrMatch(
      search_labels.data(), actual_labels.data(), params.n_query_row, MLCommon::Compare<int>()));
  }

  void testRegression()
  {
    rmm::device_uvector<int> actual_labels(params.n_query_row, stream);
    rmm::device_uvector<int> expected_labels(params.n_query_row, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(actual_labels.data(), 0, actual_labels.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(expected_labels.data(), 0, expected_labels.size() * sizeof(T), stream));

    create_data();

    brute_force_knn(handle,
                    part_inputs,
                    part_sizes,
                    params.n_cols,
                    search_data.data(),
                    params.n_query_row,
                    output_indices.data(),
                    output_dists.data(),
                    params.n_neighbors,
                    true,
                    true);

    rmm::device_uvector<float> index_labels_float(params.n_rows * params.n_parts, stream);
    rmm::device_uvector<float> query_labels_float(params.n_query_row, stream);
    to_float<<<raft::ceildiv((int)index_labels_float.size(), 32), 32, 0, stream>>>(
      index_labels_float.data(), index_labels.data(), index_labels_float.size());
    to_float<<<raft::ceildiv(params.n_query_row, 32), 32, 0, stream>>>(
      query_labels_float.data(), search_labels.data(), params.n_query_row);
    handle.sync_stream(stream);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    rmm::device_uvector<float> actual_labels_float(params.n_query_row, stream);

    vector<float*> full_labels(1);
    full_labels[0] = index_labels_float.data();

    knn_regress(handle,
                actual_labels_float.data(),
                output_indices.data(),
                full_labels,
                params.n_rows,
                params.n_query_row,
                params.n_neighbors);

    ASSERT_TRUE(MLCommon::devArrMatch(query_labels_float.data(),
                                      actual_labels_float.data(),
                                      params.n_query_row,
                                      MLCommon::Compare<float>()));
  }

 private:
  void create_data()
  {
    cudaStream_t stream = handle.get_stream().get();

    rmm::device_uvector<T> rand_centers(params.n_centers * params.n_cols, stream);
    Rng r(0, GeneratorType::GenPhilox);
    r.uniform(rand_centers.data(), params.n_centers * params.n_cols, -10.0f, 10.0f, stream);

    // Create index parts
    create_index_parts(handle,
                       index_data.data(),
                       index_labels.data(),
                       part_inputs,
                       part_labels,
                       part_sizes,
                       params,
                       rand_centers.data());

    gen_blobs(handle,
              search_data.data(),
              search_labels.data(),
              params.n_query_row,
              params.n_cols,
              params.n_centers,
              rand_centers.data());
  }

  raft::handle_t handle;
  cudaStream_t stream = 0;

  KNNInputs params;

  rmm::device_uvector<float> index_data;
  rmm::device_uvector<int> index_labels;

  vector<float*> part_inputs;
  vector<int*> part_labels;
  vector<int> part_sizes;

  rmm::device_uvector<float> search_data;
  rmm::device_uvector<int> search_labels;

  rmm::device_uvector<float> output_dists;
  rmm::device_uvector<int64_t> output_indices;
};

const std::vector<KNNInputs> inputs = {{50, 5, 2, 25, 5, 2},
                                       {50, 5, 2, 25, 10, 2},
                                       {500, 5, 2, 25, 5, 7},
                                       {500, 50, 2, 25, 10, 7},
                                       {500, 50, 7, 25, 5, 7},
                                       {50, 5, 3, 15, 5, 7}};

typedef KNNTest<float> KNNTestF;
TEST_P(KNNTestF, BruteForce) { this->testBruteForce(); }
TEST_P(KNNTestF, Classification) { this->testClassification(); }
TEST_P(KNNTestF, Regression) { this->testRegression(); }

INSTANTIATE_TEST_CASE_P(KNNTest, KNNTestF, ::testing::ValuesIn(inputs));

}  // end namespace ML
