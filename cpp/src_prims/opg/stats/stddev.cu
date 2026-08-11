/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/common/utils.hpp>
#include <cuml/prims/opg/comm_utils.h>
#include <cuml/prims/opg/stats/stddev.hpp>

#include <raft/linalg/divide.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/stats/sum.cuh>

#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace Stats {
namespace opg {

template <typename Type, typename IdxType, int TPB>
CUML_KERNEL void varsPartitionKernelColMajor(
  Type* var, const Type* data, const Type* mu, IdxType D, IdxType N, IdxType K)
{
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type thread_data = Type(0);
  IdxType colStart = N * blockIdx.x;
  Type m           = mu[blockIdx.x];
  for (IdxType i = threadIdx.x; i < N; i += TPB) {
    IdxType idx = colStart + i;
    Type diff   = data[idx] - m;
    thread_data += diff * diff;
  }
  Type acc = BlockReduce(temp_storage).Sum(thread_data);
  if (threadIdx.x == 0) { var[blockIdx.x] = acc / K; }
}

template <typename Type, typename IdxType = int>
void varPartition(
  Type* std, const Type* data, const Type* mu, IdxType D, IdxType N, IdxType K, cudaStream_t stream)
{
  static const int TPB = 256;
  varsPartitionKernelColMajor<Type, IdxType, TPB><<<D, TPB, 0, stream>>>(std, data, mu, D, N, K);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T, int TPB = 256>
void var_impl(const raft::handle_t& handle,
              Matrix::Data<T>& out,
              const std::vector<Matrix::Data<T>*>& in,
              const Matrix::PartDescriptor& inDesc,
              const T* mu,
              cudaStream_t* streams,
              int n_streams)
{
  auto& comm = handle.get_comms();

  rmm::device_uvector<T> local_vars_tmp(in.size() * inDesc.N, streams[0]);
  rmm::device_uvector<T> local_vars_tmp_t(in.size() * inDesc.N, streams[0]);

  std::vector<Matrix::RankSizePair*> localBlocks = inDesc.blocksOwnedBy(comm.get_rank());

  for (size_t i = 0; i < localBlocks.size(); i++) {
    T* loc = local_vars_tmp.data() + (i * inDesc.N);
    Stats::opg::varPartition(
      loc, in[i]->ptr, mu, inDesc.N, localBlocks[i]->size, inDesc.M, streams[i % n_streams]);
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(streams[i]));
  }

  raft::linalg::transpose(handle,
                          local_vars_tmp.data(),
                          local_vars_tmp_t.data(),
                          inDesc.N,
                          localBlocks.size(),
                          streams[0]);

  raft::stats::sum<false>(
    local_vars_tmp.data(), local_vars_tmp_t.data(), inDesc.N, localBlocks.size(), streams[0]);

  comm.allreduce(local_vars_tmp.data(), out.ptr, inDesc.N, raft::comms::op_t::SUM, streams[0]);

  comm.sync_stream(streams[0]);
}

void var(const raft::handle_t& handle,
         Matrix::Data<double>& out,
         const std::vector<Matrix::Data<double>*>& in,
         const Matrix::PartDescriptor& inDesc,
         const double* mu,
         cudaStream_t* streams,
         int n_streams)
{
  var_impl<double>(handle, out, in, inDesc, mu, streams, n_streams);
}

void var(const raft::handle_t& handle,
         Matrix::Data<float>& out,
         const std::vector<Matrix::Data<float>*>& in,
         const Matrix::PartDescriptor& inDesc,
         const float* mu,
         cudaStream_t* streams,
         int n_streams)
{
  var_impl<float>(handle, out, in, inDesc, mu, streams, n_streams);
}

};  // namespace opg
};  // namespace Stats
};  // namespace MLCommon
