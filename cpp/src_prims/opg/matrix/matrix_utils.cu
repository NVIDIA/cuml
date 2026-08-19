/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuml/common/utils.hpp>
#include <cuml/prims/opg/matrix/matrix_utils.hpp>

#include <raft/linalg/transpose.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace MLCommon {
namespace Matrix {
namespace opg {

template <typename T>
void gatherPart(const raft::handle_t& h,
                T* gatheredPart,
                std::vector<Matrix::Data<T>*>& parts,
                Matrix::PartDescriptor& desc,
                int partIndex,
                int rootRank,
                int myRank,
                cudaStream_t stream)
{
  ASSERT(partIndex < desc.partsToRanks.size(),
         "MLCommon::Matrix::opg::gatherPart: Part index is out of range");
  const auto& comm = h.get_comms();
  std::vector<raft::comms::request_t> requests;
  int items = desc.partsToRanks[partIndex]->size * desc.N;
  comm.sync_stream(stream);
  if (desc.partsToRanks[partIndex]->rank == myRank) {
    requests.resize(requests.size() + 1);
    int localIndex = 0;
    for (int i = 0; i < partIndex; i++) {
      if (desc.partsToRanks[i]->rank == myRank) { localIndex++; }
    }

    comm.isend(parts[localIndex]->ptr, items, rootRank, 0, &requests.back());
  }
  if (myRank == rootRank) {
    requests.resize(requests.size() + 1);
    comm.irecv(gatheredPart, items, desc.partsToRanks[partIndex]->rank, 0, &requests.back());
  }
  comm.waitall(requests.size(), requests.data());
}

template <typename T>
void allGatherPart(const raft::handle_t& h,
                   T* gatheredPart,
                   std::vector<Matrix::Data<T>*>& parts,
                   Matrix::PartDescriptor& desc,
                   int partIndex,
                   int myRank,
                   cudaStream_t stream)
{
  ASSERT(partIndex < desc.partsToRanks.size(),
         "MLCommon::Matrix::opg::allGatherPart: Part index is out of range");

  const auto& comm = h.get_comms();

  int partLength = desc.partsToRanks[partIndex]->size * desc.N;

  if (desc.partsToRanks[partIndex]->rank == myRank) {
    int localIndex = 0;
    for (int i = 0; i < partIndex; i++) {
      if (desc.partsToRanks[i]->rank == myRank) { localIndex++; }
    }
    raft::copy(gatheredPart, parts[localIndex]->ptr, partLength, stream);
    comm.sync_stream(stream);
  }
  comm.bcast(gatheredPart, partLength, desc.partsToRanks[partIndex]->rank, stream);
}

template <typename T>
void gather(const raft::handle_t& h,
            T* gatheredMatrix,
            std::vector<Matrix::Data<T>*>& parts,
            Matrix::PartDescriptor& desc,
            int rootRank,
            int myRank,
            cudaStream_t stream)
{
  size_t maxPartSize = 0;
  for (size_t i = 0; i < desc.partsToRanks.size(); i++) {
    if (maxPartSize < desc.partsToRanks[i]->size) { maxPartSize = desc.partsToRanks[i]->size; }
  }

  rmm::device_uvector<T> partBuffer(0, stream);
  if (myRank == rootRank) { partBuffer.resize(maxPartSize * desc.N, stream); }
  size_t offset = 0;
  for (size_t i = 0; i < desc.partsToRanks.size(); i++) {
    gatherPart(h, partBuffer.data(), parts, desc, i, rootRank, myRank, stream);

    if (myRank == rootRank) {
      if (desc.layout == Matrix::LayoutRowMajor) {
        raft::copy(
          gatheredMatrix + offset, partBuffer.data(), desc.partsToRanks[i]->size * desc.N, stream);
      } else {
        raft::linalg::transpose(h,
                                partBuffer.data(),
                                gatheredMatrix + offset,
                                (int)desc.partsToRanks[i]->size,
                                (int)desc.N,
                                stream);
      }
    }
    offset += desc.partsToRanks[i]->size * desc.N;
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename T>
void allGather(const raft::handle_t& h,
               T* gatheredMatrix,
               std::vector<Matrix::Data<T>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  size_t maxPartSize = 0;
  for (size_t i = 0; i < desc.partsToRanks.size(); i++) {
    if (maxPartSize < desc.partsToRanks[i]->size) { maxPartSize = desc.partsToRanks[i]->size; }
  }

  rmm::device_uvector<T> partBuffer(maxPartSize * desc.N, stream);
  size_t offset = 0;
  for (size_t i = 0; i < desc.partsToRanks.size(); i++) {
    allGatherPart(h, partBuffer.data(), parts, desc, i, myRank, stream);
    if (desc.layout == Matrix::LayoutRowMajor) {
      raft::copy(
        gatheredMatrix + offset, partBuffer.data(), desc.partsToRanks[i]->size * desc.N, stream);
    } else {
      raft::linalg::transpose(h,
                              partBuffer.data(),
                              gatheredMatrix + offset,
                              (int)desc.partsToRanks[i]->size,
                              (int)desc.N,
                              stream);
    }
    offset += desc.partsToRanks[i]->size * desc.N;
  }
}

template <typename T>
void allocate(const raft::handle_t& h,
              std::vector<Matrix::Data<T>*>& parts,
              Matrix::PartDescriptor& desc,
              int myRank,
              cudaStream_t stream)
{
  auto allocator = rmm::mr::get_current_device_resource_ref();
  for (size_t i = 0; i < desc.partsToRanks.size(); i++) {
    if (myRank == desc.partsToRanks[i]->rank) {
      int partSize          = desc.partsToRanks[i]->size * desc.N;
      T* partMem            = (T*)allocator.allocate(stream, partSize * sizeof(T));
      Matrix::Data<T>* part = new Matrix::Data<T>(partMem, partSize);
      parts.push_back(part);
    }
  }
  return;
}

template <typename T>
void deallocate(const raft::handle_t& h,
                std::vector<Matrix::Data<T>*>& parts,
                Matrix::PartDescriptor& desc,
                int myRank,
                cudaStream_t stream)
{
  auto allocator = rmm::mr::get_current_device_resource_ref();
  for (size_t i = 0, localIndex = 0; i < desc.partsToRanks.size(); i++) {
    if (myRank == desc.partsToRanks[i]->rank) {
      int partSize = desc.partsToRanks[i]->size * desc.N;
      allocator.deallocate(stream, parts[localIndex]->ptr, partSize * sizeof(T));
      localIndex++;
    }
  }
  for (size_t i = 0, localIndex = 0; i < desc.partsToRanks.size(); i++) {
    if (myRank == desc.partsToRanks[i]->rank) {
      delete parts[localIndex];
      localIndex++;
    }
  }
  return;
}

template <typename T>
void randomize(const raft::handle_t& h,
               raft::random::Rng& r,
               std::vector<Matrix::Data<T>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream,
               T low  = T(-1.0),
               T high = T(1.0))
{
  for (size_t i = 0, localIndex = 0; i < desc.partsToRanks.size(); i++) {
    if (myRank == desc.partsToRanks[i]->rank) {
      int partSize = desc.partsToRanks[i]->size * desc.N;
      r.uniform(parts[localIndex]->ptr, partSize, low, high, stream);
      localIndex++;
    }
  }
}

template <typename T>
void reset(const raft::handle_t& h,
           std::vector<Matrix::Data<T>*>& parts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  for (size_t i = 0, localIndex = 0; i < desc.partsToRanks.size(); i++) {
    if (myRank == desc.partsToRanks[i]->rank) {
      size_t partSize = desc.partsToRanks[i]->size * desc.N * sizeof(T);
      RAFT_CUDA_TRY(cudaMemsetAsync(parts[localIndex]->ptr, 0, partSize, stream));
      localIndex++;
    }
  }
}

template <typename T>
CUML_KERNEL void printRaw2DKernel(T* buffer, int rows, int cols, bool isColMajor)
{
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      if (isColMajor) {
        printf("%8.6f ", buffer[c * rows + r]);
      } else {
        printf("%8.6f ", buffer[r * cols + c]);
      }
    }
    printf("\n");
  }
  return;
}

template <typename T>
void printRaw2D(T* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream)
{
  printRaw2DKernel<<<1, 1, 0, stream>>>(buffer, rows, cols, isColMajor);
}

template <typename T>
void print(const raft::handle_t& h,
           std::vector<Matrix::Data<T>*>& parts,
           Matrix::PartDescriptor& desc,
           const char* matrixName,
           int myRank,
           cudaStream_t stream)
{
  rmm::device_uvector<T> buffer(desc.M * desc.N, stream);
  if (myRank == 0) { buffer.resize(desc.M * desc.N, stream); }
  Matrix::opg::gather(h, buffer.data(), parts, desc, 0, myRank, stream);
  if (myRank == 0) {
    printf("%s = [", matrixName);
    printRaw2D(buffer.data(), desc.M, desc.N, false, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    printf("];\n#-------------------------------\n");
  }
}

// Instantiations

void gatherPart(const raft::handle_t& h,
                float* gatheredPart,
                std::vector<Matrix::Data<float>*>& parts,
                Matrix::PartDescriptor& desc,
                int partIndex,
                int rootRank,
                int myRank,
                cudaStream_t stream)
{
  gatherPart<float>(h, gatheredPart, parts, desc, partIndex, rootRank, myRank, stream);
}

void allGatherPart(const raft::handle_t& h,
                   float* gatheredPart,
                   std::vector<Matrix::Data<float>*>& parts,
                   Matrix::PartDescriptor& desc,
                   int partIndex,
                   int myRank,
                   cudaStream_t stream)
{
  allGatherPart<float>(h, gatheredPart, parts, desc, partIndex, myRank, stream);
}

void gather(const raft::handle_t& h,
            float* gatheredMatrix,
            std::vector<Matrix::Data<float>*>& parts,
            Matrix::PartDescriptor& desc,
            int rootRank,
            int myRank,
            cudaStream_t stream)
{
  gather<float>(h, gatheredMatrix, parts, desc, rootRank, myRank, stream);
}

void allGather(const raft::handle_t& h,
               float* gatheredMatrix,
               std::vector<Matrix::Data<float>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  allGather<float>(h, gatheredMatrix, parts, desc, myRank, stream);
}

void allocate(const raft::handle_t& h,
              std::vector<Matrix::Data<float>*>& parts,
              Matrix::PartDescriptor& desc,
              int myRank,
              cudaStream_t stream)
{
  allocate<float>(h, parts, desc, myRank, stream);
  return;
}

void deallocate(const raft::handle_t& h,
                std::vector<Matrix::Data<float>*>& parts,
                Matrix::PartDescriptor& desc,
                int myRank,
                cudaStream_t stream)
{
  deallocate<float>(h, parts, desc, myRank, stream);
}

void randomize(const raft::handle_t& h,
               raft::random::Rng& r,
               std::vector<Matrix::Data<float>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream,
               float low,
               float high)
{
  randomize<float>(h, r, parts, desc, myRank, stream, low, high);
}

void reset(const raft::handle_t& h,
           std::vector<Matrix::Data<float>*>& parts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  reset<float>(h, parts, desc, myRank, stream);
}

void printRaw2D(float* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream)
{
  printRaw2D<float>(buffer, rows, cols, isColMajor, stream);
}

void print(const raft::handle_t& h,
           std::vector<Matrix::Data<float>*>& parts,
           Matrix::PartDescriptor& desc,
           const char* matrixName,
           int myRank,
           cudaStream_t stream)
{
  print<float>(h, parts, desc, matrixName, myRank, stream);
}

void gatherPart(const raft::handle_t& h,
                double* gatheredPart,
                std::vector<Matrix::Data<double>*>& parts,
                Matrix::PartDescriptor& desc,
                int partIndex,
                int rootRank,
                int myRank,
                cudaStream_t stream)
{
  gatherPart<double>(h, gatheredPart, parts, desc, partIndex, rootRank, myRank, stream);
}

void allGatherPart(const raft::handle_t& h,
                   double* gatheredPart,
                   std::vector<Matrix::Data<double>*>& parts,
                   Matrix::PartDescriptor& desc,
                   int partIndex,
                   int myRank,
                   cudaStream_t stream)
{
  allGatherPart<double>(h, gatheredPart, parts, desc, partIndex, myRank, stream);
}

void gather(const raft::handle_t& h,
            double* gatheredMatrix,
            std::vector<Matrix::Data<double>*>& parts,
            Matrix::PartDescriptor& desc,
            int rootRank,
            int myRank,
            cudaStream_t stream)
{
  gather<double>(h, gatheredMatrix, parts, desc, rootRank, myRank, stream);
}

void allGather(const raft::handle_t& h,
               double* gatheredMatrix,
               std::vector<Matrix::Data<double>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream)
{
  allGather<double>(h, gatheredMatrix, parts, desc, myRank, stream);
}

void allocate(const raft::handle_t& h,
              std::vector<Matrix::Data<double>*>& parts,
              Matrix::PartDescriptor& desc,
              int myRank,
              cudaStream_t stream)
{
  allocate<double>(h, parts, desc, myRank, stream);
}

void deallocate(const raft::handle_t& h,
                std::vector<Matrix::Data<double>*>& parts,
                Matrix::PartDescriptor& desc,
                int myRank,
                cudaStream_t stream)
{
  deallocate<double>(h, parts, desc, myRank, stream);
}

void randomize(const raft::handle_t& h,
               raft::random::Rng& r,
               std::vector<Matrix::Data<double>*>& parts,
               Matrix::PartDescriptor& desc,
               int myRank,
               cudaStream_t stream,
               double low,
               double high)
{
  randomize<double>(h, r, parts, desc, myRank, stream, low, high);
}

void reset(const raft::handle_t& h,
           std::vector<Matrix::Data<double>*>& parts,
           Matrix::PartDescriptor& desc,
           int myRank,
           cudaStream_t stream)
{
  reset<double>(h, parts, desc, myRank, stream);
}

void printRaw2D(double* buffer, int rows, int cols, bool isColMajor, cudaStream_t stream)
{
  printRaw2D<double>(buffer, rows, cols, isColMajor, stream);
}

void print(const raft::handle_t& h,
           std::vector<Matrix::Data<double>*>& parts,
           Matrix::PartDescriptor& desc,
           const char* matrixName,
           int myRank,
           cudaStream_t stream)
{
  print<double>(h, parts, desc, matrixName, myRank, stream);
}

}  // end namespace opg
}  // end namespace Matrix
}  // end namespace MLCommon
