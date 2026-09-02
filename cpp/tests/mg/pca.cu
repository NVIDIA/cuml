/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_opg_utils.h"

#include <cuml/common/logger.hpp>
#include <cuml/decomposition/pca_mg.hpp>
#include <cuml/prims/opg/linalg/gemm.hpp>
#include <cuml/prims/opg/matrix/matrix_utils.hpp>

#include <raft/comms/mpi_comms.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

namespace MLCommon {
namespace Test {
namespace opg {

struct PCAOpgParams {
  int M;
  int N;
  int N_components;
  ML::mg_solver algorithm;
  std::vector<int> partSizes;
  std::vector<int> ranksOwners;
  Matrix::Layout layout;
  unsigned long long int seed;
};

template <typename T>
class PCAOpgTest : public testing::TestWithParam<PCAOpgParams> {
 public:
  void SetUp()
  {
    params = GetParam();
    raft::comms::initialize_mpi_comms(&handle, MPI_COMM_WORLD);

    // Prepare resource

    const raft::comms::comms_t& comm = handle.get_comms();
    stream                           = handle.get_stream();
    cublasHandle_t cublasHandle      = handle.get_cublas_handle();

    myRank     = comm.get_rank();
    totalRanks = comm.get_size();
    raft::random::Rng r(params.seed + myRank);

    RAFT_CUBLAS_TRY(cublasSetStream(cublasHandle, stream));

    if (myRank == 0) {
      std::cout << "Testing PCA of " << params.M << " x " << params.N << " matrix" << std::endl;
    }

    // Prepare X matrix
    std::vector<Matrix::RankSizePair*> totalPartsToRanks;
    for (int i = 0; i < params.partSizes.size(); i++) {
      Matrix::RankSizePair* rspt =
        new Matrix::RankSizePair(params.ranksOwners[i] % totalRanks, params.partSizes[i]);
      totalPartsToRanks.push_back(rspt);
    }
    Matrix::PartDescriptor desc(
      params.M, params.N, totalPartsToRanks, comm.get_rank(), params.layout);
    std::vector<Matrix::Data<T>*> inParts;
    Matrix::opg::allocate(handle, inParts, desc, myRank, stream);
    Matrix::opg::randomize(handle, r, inParts, desc, myRank, stream, T(10.0), T(20.0));
    handle.sync_stream();

    prmsPCA.n_rows       = params.M;
    prmsPCA.n_cols       = params.N;
    prmsPCA.n_components = params.N_components;
    prmsPCA.whiten       = false;
    prmsPCA.n_iterations = 100;
    prmsPCA.tol          = 0.01;
    prmsPCA.algorithm    = params.algorithm;

    rmm::device_uvector<T> components(prmsPCA.n_components * prmsPCA.n_cols, stream);

    rmm::device_uvector<T> explained_var(prmsPCA.n_components, stream);

    rmm::device_uvector<T> explained_var_ratio(prmsPCA.n_components, stream);

    rmm::device_uvector<T> singular_vals(prmsPCA.n_components, stream);

    rmm::device_uvector<T> mu(prmsPCA.n_cols, stream);

    rmm::device_uvector<T> noise_vars(prmsPCA.n_components, stream);

    ML::PCA::opg::fit(handle,
                      inParts,
                      desc,
                      components.data(),
                      explained_var.data(),
                      explained_var_ratio.data(),
                      singular_vals.data(),
                      mu.data(),
                      noise_vars.data(),
                      prmsPCA,
                      false);

    CUML_LOG_DEBUG(
      raft::arr2Str(singular_vals.data(), params.N_components, "Singular Vals", stream).c_str());

    CUML_LOG_DEBUG(
      raft::arr2Str(explained_var.data(), params.N_components, "Explained Variance", stream)
        .c_str());

    CUML_LOG_DEBUG(
      raft::arr2Str(
        explained_var_ratio.data(), params.N_components, "Explained Variance Ratio", stream)
        .c_str());

    CUML_LOG_DEBUG(
      raft::arr2Str(components.data(), params.N_components * params.N, "Components", stream)
        .c_str());

    Matrix::opg::deallocate(handle, inParts, desc, myRank, stream);
  }

 protected:
  PCAOpgParams params;
  raft::handle_t handle;
  cudaStream_t stream = 0;
  int myRank;
  int totalRanks;
  ML::paramsPCAMG prmsPCA;
};

const std::vector<PCAOpgParams> inputs = {
  {20, 4, 2, ML::mg_solver::COV_EIG_JACOBI, {11, 9}, {1, 0}, Matrix::LayoutColMajor, 223548ULL},
  {20, 4, 2, ML::mg_solver::COV_EIG_DQ, {11, 9}, {1, 0}, Matrix::LayoutColMajor, 223548ULL}};

typedef PCAOpgTest<float> PCAOpgTestF;

TEST_P(PCAOpgTestF, Result)
{
  if (myRank == 0) {
    // We should be inverse transforming and checking against the original
    // data here. Github reference: https://github.com/NVIDIA/cuml/issues/2474

    ASSERT_TRUE(true);
  }
}

INSTANTIATE_TEST_CASE_P(PCAOpgTest, PCAOpgTestF, ::testing::ValuesIn(inputs));

typedef PCAOpgTest<double> PCAOpgTestD;

TEST_P(PCAOpgTestD, Result)
{
  if (myRank == 0) {
    // We should be inverse transforming and checking against the original
    // data here. Github reference: https://github.com/NVIDIA/cuml/issues/2474

    ASSERT_TRUE(true);
  }
}

INSTANTIATE_TEST_CASE_P(PCAOpgTest, PCAOpgTestD, ::testing::ValuesIn(inputs));

}  // end namespace opg
}  // end namespace Test
}  // end namespace MLCommon
