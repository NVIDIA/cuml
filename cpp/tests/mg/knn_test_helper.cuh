/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../prims/test_utils.h"
#include "test_opg_utils.h"

#include <cuml/neighbors/knn_mg.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>
#include <selection/knn.cuh>

#include <memory>
#include <optional>

namespace ML {
namespace KNN {
namespace opg {

namespace Matrix = MLCommon::Matrix;

struct KNNParams {
  int k;
  int n_outputs;
  int n_classes;

  size_t min_rows;
  size_t n_cols;

  int n_query_parts;
  int n_index_parts;

  size_t batch_size;
};

template <typename T>
void generate_partitions(float* data,
                         T* outputs,
                         size_t n_rows,
                         int n_cols,
                         int n_clusters,
                         int my_rank,
                         cudaStream_t stream);

template <typename T>
class KNNTestHelper {
 public:
  void generate_data(const KNNParams& params)
  {
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);
    const auto& comm = handle.get_comms();

    this->stream = handle.get_stream();

    int my_rank = comm.get_rank();
    int size    = comm.get_size();

    this->index_parts_per_rank = raft::ceildiv(params.n_index_parts, size);
    this->query_parts_per_rank = raft::ceildiv(params.n_query_parts, size);

    for (int cur_rank = 0; cur_rank < size; cur_rank++) {
      int ippr = this->index_parts_per_rank;
      int qppr = this->query_parts_per_rank;
      if (cur_rank == size - 1) {
        ippr = params.n_index_parts - (cur_rank * this->index_parts_per_rank);
        qppr = params.n_query_parts - (cur_rank * this->query_parts_per_rank);
      }

      for (int part_n = 0; part_n < ippr; part_n++) {
        Matrix::RankSizePair* rsp = new Matrix::RankSizePair(cur_rank, params.min_rows);
        this->idxPartsToRanks.push_back(rsp);
      }

      for (int part_n = 0; part_n < qppr; part_n++) {
        Matrix::RankSizePair* rsp = new Matrix::RankSizePair(cur_rank, params.min_rows);
        this->queryPartsToRanks.push_back(rsp);
      }
    }

    this->idx_desc = new Matrix::PartDescriptor(params.min_rows * params.n_index_parts,
                                                params.n_cols,
                                                this->idxPartsToRanks,
                                                comm.get_rank());

    this->query_desc = new Matrix::PartDescriptor(params.min_rows * params.n_query_parts,
                                                  params.n_cols,
                                                  this->queryPartsToRanks,
                                                  comm.get_rank());

    if (my_rank == size - 1) {
      this->index_parts_per_rank = params.n_index_parts - ((size - 1) * this->index_parts_per_rank);
      query_parts_per_rank       = params.n_query_parts - ((size - 1) * query_parts_per_rank);
    }

    auto total_rows = static_cast<int64_t>(
      (this->index_parts_per_rank + this->query_parts_per_rank) * params.min_rows);
    auto total_data_elements =
      static_cast<int64_t>(total_rows * static_cast<int64_t>(params.n_cols));

    this->data_storage.emplace(
      raft::make_device_vector<float, int64_t>(handle, total_data_elements));
    this->label_storage.emplace(raft::make_device_vector<T, int64_t>(handle, total_rows));

    float* data = this->data_storage->data_handle();
    T* labels   = this->label_storage->data_handle();

    generate_partitions<T>(
      data,
      labels,
      (this->index_parts_per_rank + this->query_parts_per_rank) * params.min_rows,
      params.n_cols,
      params.n_classes,
      my_rank,
      this->stream.get());

    y.resize(this->index_parts_per_rank);
    for (int i = 0; i < this->index_parts_per_rank; i++) {
      Matrix::Data<float>* i_d = new Matrix::Data<float>(
        data + (i * params.min_rows * params.n_cols), params.min_rows * params.n_cols);
      this->index_parts.push_back(i_d);

      for (int j = 0; j < params.n_outputs; j++) {
        y[i].push_back(labels + (i * params.min_rows));
      }
    }

    int end_of_idx = this->index_parts_per_rank * params.min_rows * params.n_cols;

    this->out_part_storage.reserve(this->query_parts_per_rank);
    this->out_i_part_storage.reserve(this->query_parts_per_rank);
    this->out_d_part_storage.reserve(this->query_parts_per_rank);

    for (int i = 0; i < query_parts_per_rank; i++) {
      Matrix::Data<float>* query_d = new Matrix::Data<float>(
        data + end_of_idx + (i * params.min_rows * params.n_cols), params.min_rows * params.n_cols);

      this->out_part_storage.emplace_back(raft::make_device_vector<T, int64_t>(
        handle, static_cast<int64_t>(params.min_rows * params.n_outputs)));
      this->out_d_part_storage.emplace_back(raft::make_device_vector<float, int64_t>(
        handle, static_cast<int64_t>(params.min_rows * params.k)));
      this->out_i_part_storage.emplace_back(raft::make_device_vector<int64_t, int64_t>(
        handle, static_cast<int64_t>(params.min_rows * params.k)));

      T* o         = this->out_part_storage.back().data_handle();
      float* d     = this->out_d_part_storage.back().data_handle();
      int64_t* ind = this->out_i_part_storage.back().data_handle();

      Matrix::Data<T>* out = new Matrix::Data<T>(o, params.min_rows * params.n_outputs);

      Matrix::floatData_t* out_d = new Matrix::floatData_t(d, params.min_rows * params.k);

      Matrix::Data<int64_t>* out_i = new Matrix::Data<int64_t>(ind, params.min_rows * params.k);

      this->query_parts.push_back(query_d);
      this->out_parts.push_back(out);
      this->out_d_parts.push_back(out_d);
      this->out_i_parts.push_back(out_i);
    }

    handle.sync_stream(stream);
  }

  void display_results()
  {
    handle.sync_stream(stream);

    std::cout << "Finished!" << std::endl;

    std::cout << raft::arr2Str(out_parts[0]->ptr, 10, "final_out", stream.get()) << std::endl;
  }

  void release_ressources(const KNNParams& params)
  {
    (void)params;
    delete this->idx_desc;
    delete this->query_desc;
    this->idx_desc   = nullptr;
    this->query_desc = nullptr;

    for (Matrix::floatData_t* fd : query_parts) {
      delete fd;
    }

    for (Matrix::floatData_t* fd : index_parts) {
      delete fd;
    }

    for (Matrix::Data<T>* fd : out_parts) {
      delete fd;
    }

    for (Matrix::Data<int64_t>* fd : out_i_parts) {
      delete fd;
    }

    for (Matrix::floatData_t* fd : out_d_parts) {
      delete fd;
    }

    for (Matrix::RankSizePair* rsp : this->queryPartsToRanks) {
      delete rsp;
    }

    for (Matrix::RankSizePair* rsp : this->idxPartsToRanks) {
      delete rsp;
    }

    this->out_parts.clear();
    this->out_i_parts.clear();
    this->out_d_parts.clear();
    this->index_parts.clear();
    this->query_parts.clear();
    this->y.clear();
    this->idxPartsToRanks.clear();
    this->queryPartsToRanks.clear();
    this->out_part_storage.clear();
    this->out_i_part_storage.clear();
    this->out_d_part_storage.clear();
    this->label_storage.reset();
    this->data_storage.reset();
  }

  raft::handle_t handle;
  std::vector<Matrix::Data<T>*> out_parts;
  std::vector<Matrix::Data<int64_t>*> out_i_parts;
  std::vector<Matrix::floatData_t*> out_d_parts;
  std::vector<Matrix::floatData_t*> index_parts;
  Matrix::PartDescriptor* idx_desc = nullptr;
  std::vector<Matrix::floatData_t*> query_parts;
  Matrix::PartDescriptor* query_desc = nullptr;
  std::vector<std::vector<T*>> y;

  cuda::stream_ref stream{cudaStream_t{nullptr}};

 private:
  int index_parts_per_rank;
  int query_parts_per_rank;
  std::vector<Matrix::RankSizePair*> idxPartsToRanks;
  std::vector<Matrix::RankSizePair*> queryPartsToRanks;

  std::optional<raft::device_vector<float, int64_t>> data_storage;
  std::optional<raft::device_vector<T, int64_t>> label_storage;
  std::vector<raft::device_vector<T, int64_t>> out_part_storage;
  std::vector<raft::device_vector<int64_t, int64_t>> out_i_part_storage;
  std::vector<raft::device_vector<float, int64_t>> out_d_part_storage;
};

}  // namespace opg
}  // namespace KNN
}  // namespace ML
