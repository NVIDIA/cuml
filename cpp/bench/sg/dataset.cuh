/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/datasets/make_blobs.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/make_regression.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda/stream>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ML {
namespace Bench {

/**
 * Indicates the dataset size. This is supposed to be used as the base class
 * by every Benchmark's Params structure.
 */
struct DatasetParams {
  /** number of rows in the dataset */
  int nrows;
  /** number of cols in the dataset */
  int ncols;
  /** number of classes in the dataset (useless for regression cases) */
  int nclasses;
  /** input dataset is stored row or col major? */
  bool rowMajor;
};

/** Holds params needed to generate blobs dataset */
struct BlobsParams {
  double cluster_std;
  bool shuffle;
  double center_box_min, center_box_max;
  uint64_t seed;
};

/** Holds params needed to generate regression dataset */
struct RegressionParams {
  int n_informative;
  int effective_rank;
  double bias;
  double tail_strength;
  double noise;
  bool shuffle;
  uint64_t seed;
};

/**
 * @brief A simple object to hold the loaded dataset for benchmarking
 * @tparam D type of the dataset (type of X)
 * @tparam L type of the labels/output (type of y)
 * @tparam IdxT type of indices
 */
template <typename D, typename L, typename IdxT = int>
struct Dataset {
  Dataset()
    : X(0, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}),
      y(0, cuda::stream_ref{cudaStream_t{cudaStreamDefault}})
  {
  }
  /** input data */
  rmm::device_uvector<D> X;
  /** labels or output associated with each row of input data */
  rmm::device_uvector<L> y;

  /** allocate space needed for the dataset */
  void allocate(const raft::handle_t& handle, const DatasetParams& p)
  {
    auto stream = handle.get_stream();
    X.resize(p.nrows * p.ncols, stream);
    y.resize(p.nrows, stream);
  }

  /** free-up the buffers */
  void deallocate(const raft::handle_t& handle, const DatasetParams& p)
  {
    auto stream = handle.get_stream();
    X.release();
    y.release();
  }

  /** whether the current dataset is for classification or regression */
  bool isClassification() const { return typeid(D) != typeid(L); }

  /**
   * Generate random blobs data. Args are the same as in make_blobs.
   * Assumes that the user has already called `allocate`
   */
  void blobs(const raft::handle_t& handle, const DatasetParams& p, const BlobsParams& b)
  {
    const auto& handle_impl = handle;
    auto stream             = handle_impl.get_stream().get();
    auto cublas_handle      = handle_impl.get_cublas_handle();

    // Make blobs will generate labels of type IdxT which has to be an integer
    // type. We cast it to a different output type if needed.
    IdxT* tmpY;
    rmm::device_uvector<IdxT> tmpY_vec(0, stream);
    if (std::is_same<L, IdxT>::value) {
      tmpY = (IdxT*)y.data();
    } else {
      tmpY_vec.resize(p.nrows, stream);
      tmpY = tmpY_vec.data();
    }

    ML::Datasets::make_blobs(handle,
                             X.data(),
                             tmpY,
                             p.nrows,
                             p.ncols,
                             p.nclasses,
                             p.rowMajor,
                             nullptr,
                             nullptr,
                             D(b.cluster_std),
                             b.shuffle,
                             D(b.center_box_min),
                             D(b.center_box_max),
                             b.seed);
    if (!std::is_same<L, IdxT>::value) {
      raft::linalg::unaryOp(
        y.data(), tmpY, p.nrows, [] __device__(IdxT z) { return (L)z; }, stream);
    }
  }

  /**
   * Generate random regression data. Args are the same as in make_regression.
   * Assumes that the user has already called `allocate`
   */
  void regression(const raft::handle_t& handle, const DatasetParams& p, const RegressionParams& r)
  {
    ASSERT(!isClassification(), "make_regression: is only for regression problems!");
    const auto& handle_impl = handle;
    auto stream             = handle_impl.get_stream().get();
    auto cublas_handle      = handle_impl.get_cublas_handle();
    auto cusolver_handle    = handle_impl.get_cusolver_dn_handle();

    D* tmpX = X.data();
    rmm::device_uvector<D> tmpX_vec(0, stream);
    if (!p.rowMajor) {
      tmpX_vec.resize(p.nrows * p.ncols, stream);
      tmpX = tmpX_vec.data();
    }
    raft::random::make_regression(handle,
                                  tmpX,
                                  y.data(),
                                  p.nrows,
                                  p.ncols,
                                  r.n_informative,
                                  stream,
                                  (D*)nullptr,
                                  1,
                                  D(r.bias),
                                  r.effective_rank,
                                  D(r.tail_strength),
                                  D(r.noise),
                                  r.shuffle,
                                  r.seed);
    if (!p.rowMajor) { raft::linalg::transpose(handle, tmpX, X.data(), p.nrows, p.ncols, stream); }
  }

  /**
   * @brief Read the input csv file and construct the dataset.
   *        Assumes that the user has already called `allocate`
   * @tparam Lambda lambda to customize how to read the data and labels
   * @param handle cuml handle
   * @param csvfile the csv file
   * @param p dataset parameters
   * @param readOp functor/operator to take the current row of values and update
   *               the dataset accordingly. Its signature is expected to be:
   * `void readOp(const std::vector<std::string>& row, std::vector<D>& X,
   *              std::vector<L>& y, int lineNum, const DatasetParams& p);`
   */
  template <typename Lambda>
  void read_csv(const raft::handle_t& handle,
                const std::string& csvfile,
                const DatasetParams& p,
                Lambda readOp)
  {
    if (isClassification() && p.nclasses <= 0) {
      ASSERT(false, "read_csv: for classification data 'nclasses' is mandatory!");
    }
    std::vector<D> _X(p.nrows * p.ncols);
    std::vector<L> _y(p.nrows);
    std::ifstream myfile;
    myfile.open(csvfile);
    std::string line;
    int counter   = 0;
    int break_cnt = p.nrows;
    while (getline(myfile, line) && (counter < p.nrows)) {
      auto row = split(line, ',');
      readOp(row, _X, _y, counter, p);
      counter++;
    }
    myfile.close();
    auto stream = handle.get_stream();
    raft::copy(X, &(_X[0]), p.nrows * p.ncols, stream);
    raft::copy(y, &(_y[0]), p.nrows, stream);
  }

 private:
  std::vector<std::string> split(const std::string& str, char delimiter)
  {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream iss(str);
    while (std::getline(iss, token, delimiter)) {
      tokens.push_back(token);
    }
    return tokens;
  }
};

namespace {
std::ostream& operator<<(std::ostream& os, const DatasetParams& d)
{
  os << "/" << d.nrows << "x" << d.ncols;
  os << "/" << (d.rowMajor ? "row" : "col");
  return os;
}
}  // namespace

}  // end namespace Bench
}  // end namespace ML
