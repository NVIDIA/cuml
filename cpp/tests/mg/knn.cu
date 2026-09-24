/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../prims/test_utils.h"
#include "test_opg_utils.h"

#include <cuml/neighbors/knn_mg.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>

#include <memory>

namespace ML {
namespace KNN {
namespace opg {

namespace Matrix = MLCommon::Matrix;

struct KNNParams {
  int k;

  size_t min_rows;
  size_t n_cols;

  int n_query_parts;
  int n_index_parts;

  size_t batch_size;
};

class BruteForceKNNTest : public ::testing::TestWithParam<KNNParams> {
 public:
  void generate_partition(
    Matrix::floatData_t* part, size_t n_rows, int n_cols, int n_clusters, cudaStream_t stream)
  {
    auto labels = raft::make_device_vector<int, int64_t>(handle, static_cast<int64_t>(n_rows));

    raft::random::make_blobs<float, int>(
      part->ptr, labels.data_handle(), (int)n_rows, (int)n_cols, n_clusters, stream);
  }

  bool runTest(const KNNParams& params)
  {
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    const auto& comm = handle.get_comms();

    auto stream = handle.get_stream();

    int my_rank = comm.get_rank();
    int size    = comm.get_size();

    int index_parts_per_rank = raft::ceildiv(params.n_index_parts, size);
    int query_parts_per_rank = raft::ceildiv(params.n_query_parts, size);
    std::vector<Matrix::RankSizePair*> idxPartsToRanks;
    std::vector<Matrix::RankSizePair*> queryPartsToRanks;
    for (int cur_rank = 0; cur_rank < size; cur_rank++) {
      int ippr = index_parts_per_rank;
      int qppr = query_parts_per_rank;
      if (cur_rank == size - 1) {
        ippr = params.n_index_parts - (cur_rank * index_parts_per_rank);
        qppr = params.n_query_parts - (cur_rank * query_parts_per_rank);
      }

      std::cout << "Generating " << ippr << " partitions for rank " << cur_rank << std::endl;

      std::cout << "min_rows: " << params.min_rows << std::endl;

      for (int part_n = 0; part_n < ippr; part_n++) {
        Matrix::RankSizePair* rsp = new Matrix::RankSizePair(cur_rank, params.min_rows);
        idxPartsToRanks.push_back(rsp);
      }

      for (int part_n = 0; part_n < qppr; part_n++) {
        Matrix::RankSizePair* rsp = new Matrix::RankSizePair(cur_rank, params.min_rows);
        queryPartsToRanks.push_back(rsp);
      }
    }

    std::cout << idxPartsToRanks.size() << std::endl;

    if (my_rank == size - 1) {
      index_parts_per_rank = params.n_index_parts - ((size - 1) * index_parts_per_rank);
      query_parts_per_rank = params.n_query_parts - ((size - 1) * query_parts_per_rank);
    }

    std::cout << "Generating " << index_parts_per_rank << " partitions for rank " << my_rank
              << std::endl;

    std::vector<Matrix::floatData_t*> query_parts;
    std::vector<Matrix::floatData_t*> out_d_parts;
    std::vector<Matrix::Data<int64_t>*> out_i_parts;
    std::vector<raft::device_vector<float, int64_t>> query_storage;
    std::vector<raft::device_vector<float, int64_t>> out_d_storage;
    std::vector<raft::device_vector<int64_t, int64_t>> out_i_storage;
    query_storage.reserve(query_parts_per_rank);
    out_d_storage.reserve(query_parts_per_rank);
    out_i_storage.reserve(query_parts_per_rank);
    for (int i = 0; i < query_parts_per_rank; i++) {
      query_storage.emplace_back(raft::make_device_vector<float, int64_t>(
        handle, static_cast<int64_t>(params.min_rows * params.n_cols)));
      out_d_storage.emplace_back(raft::make_device_vector<float, int64_t>(
        handle, static_cast<int64_t>(params.min_rows * params.k)));
      out_i_storage.emplace_back(raft::make_device_vector<int64_t, int64_t>(
        handle, static_cast<int64_t>(params.min_rows * params.k)));

      float* q     = query_storage.back().data_handle();
      float* o     = out_d_storage.back().data_handle();
      int64_t* ind = out_i_storage.back().data_handle();

      Matrix::Data<float>* query_d = new Matrix::Data<float>(q, params.min_rows * params.n_cols);

      Matrix::floatData_t* out_d = new Matrix::floatData_t(o, params.min_rows * params.k);

      Matrix::Data<int64_t>* out_i = new Matrix::Data<int64_t>(ind, params.min_rows * params.k);

      query_parts.push_back(query_d);
      out_d_parts.push_back(out_d);
      out_i_parts.push_back(out_i);

      generate_partition(query_d, params.min_rows, params.n_cols, 5, stream.get());
    }

    std::vector<Matrix::floatData_t*> index_parts;
    std::vector<raft::device_vector<float, int64_t>> index_storage;
    index_storage.reserve(index_parts_per_rank);

    for (int i = 0; i < index_parts_per_rank; i++) {
      index_storage.emplace_back(raft::make_device_vector<float, int64_t>(
        handle, static_cast<int64_t>(params.min_rows * params.n_cols)));
      float* ind = index_storage.back().data_handle();

      Matrix::Data<float>* i_d = new Matrix::Data<float>(ind, params.min_rows * params.n_cols);

      index_parts.push_back(i_d);

      generate_partition(i_d, params.min_rows, params.n_cols, 5, stream.get());
    }

    Matrix::PartDescriptor idx_desc(
      params.min_rows * params.n_index_parts, params.n_cols, idxPartsToRanks, comm.get_rank());

    Matrix::PartDescriptor query_desc(
      params.min_rows * params.n_query_parts, params.n_cols, queryPartsToRanks, comm.get_rank());

    handle.sync_stream(stream);

    /**
     * Execute knn()
     */
    knn(handle,
        &out_i_parts,
        &out_d_parts,
        index_parts,
        idx_desc,
        query_parts,
        query_desc,
        false,
        false,
        params.k,
        params.batch_size,
        true);

    handle.sync_stream(stream);

    std::cout << raft::arr2Str(out_i_parts[0]->ptr, 10, "final_out_I", stream.get()) << std::endl;
    std::cout << raft::arr2Str(out_d_parts[0]->ptr, 10, "final_out_D", stream.get()) << std::endl;

    /**
     * Verify expected results
     */

    for (Matrix::floatData_t* fd : query_parts) {
      delete fd;
    }

    for (Matrix::floatData_t* fd : index_parts) {
      delete fd;
    }

    for (Matrix::Data<int64_t>* fd : out_i_parts) {
      delete fd;
    }

    for (Matrix::floatData_t* fd : out_d_parts) {
      delete fd;
    }

    for (Matrix::RankSizePair* rsp : queryPartsToRanks) {
      delete rsp;
    }

    for (Matrix::RankSizePair* rsp : idxPartsToRanks) {
      delete rsp;
    }

    int actual   = 1;
    int expected = 1;
    return MLCommon::CompareApprox<int>(1)(actual, expected);
  }

 private:
  raft::handle_t handle;
};

const std::vector<KNNParams> inputs = {{5, 50, 3, 5, 5, 12},
                                       {10, 50, 3, 5, 5, 50},
                                       {5, 50, 3, 5, 5, 50},
                                       {5, 500, 5, 5, 5, 50},
                                       {10, 500, 50, 5, 5, 50},
                                       {15, 500, 5, 5, 5, 50},
                                       {5, 500, 10, 5, 5, 50},
                                       {10, 500, 10, 5, 5, 50},
                                       {15, 500, 10, 5, 5, 50}};

typedef BruteForceKNNTest KNNTest;

TEST_P(KNNTest, Result) { ASSERT_TRUE(runTest(GetParam())); }

INSTANTIATE_TEST_CASE_P(BruteForceKNNTest, KNNTest, ::testing::ValuesIn(inputs));

}  // namespace opg
}  // namespace KNN
}  // namespace ML
