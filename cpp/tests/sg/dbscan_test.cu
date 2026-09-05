/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/distance_type.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <vector>

namespace ML {

using namespace Datasets;
using namespace Metrics;
using namespace std;

// Note: false negatives are theoretically possible, given that border
// points are ambiguous.
// If test failures are observed, these tests might need to be re-written
// (cf how the Python tests work).

template <typename T, typename IdxT>
struct DbscanInputs {
  IdxT n_row;
  IdxT n_col;
  IdxT n_centers;
  T cluster_std;
  T eps;
  int min_pts;
  size_t max_bytes_per_batch;
  unsigned long long int seed;
  ML::distance::DistanceType metric;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const DbscanInputs<T, IdxT>& dims)
{
  return os;
}

template <typename T, typename IdxT>
class DbscanTest : public ::testing::TestWithParam<DbscanInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;
    auto stream = handle.get_stream();

    params = ::testing::TestWithParam<DbscanInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<T> out(params.n_row * params.n_col, stream);
    rmm::device_uvector<IdxT> l(params.n_row, stream);
    rmm::device_uvector<T> dist(
      params.metric == ML::distance::DistanceType::Precomputed ? params.n_row * params.n_row : 0,
      stream);

    make_blobs(handle,
               out.data(),
               l.data(),
               params.n_row,
               params.n_col,
               params.n_centers,
               true,
               nullptr,
               nullptr,
               params.cluster_std,
               true,
               -10.0f,
               10.0f,
               params.seed);

    if (params.metric == ML::distance::DistanceType::Precomputed) {
      ML::Metrics::pairwise_distance(handle,
                                     out.data(),
                                     out.data(),
                                     dist.data(),
                                     params.n_row,
                                     params.n_row,
                                     params.n_col,
                                     ML::distance::DistanceType::L2SqrtUnexpanded);
    }

    rmm::device_uvector<IdxT> labels(params.n_row, stream);
    rmm::device_uvector<IdxT> labels_ref(params.n_row, stream);

    raft::copy(labels_ref.data(), l.data(), params.n_row, stream);

    handle.sync_stream(stream);

    Dbscan::fit(handle,
                params.metric == ML::distance::DistanceType::Precomputed ? dist.data() : out.data(),
                params.n_row,
                params.n_col,
                params.eps,
                params.min_pts,
                params.metric,
                labels.data(),
                nullptr,
                nullptr,
                params.max_bytes_per_batch);

    handle.sync_stream(stream);

    score = adjusted_rand_index(handle, labels_ref.data(), labels.data(), params.n_row);

    if (score < 1.0) {
      auto str = raft::arr2Str(labels_ref.data(), params.n_row, "labels_ref", handle.get_stream().get());
      CUML_LOG_DEBUG("y: %s", str.c_str());
      str = raft::arr2Str(labels.data(), params.n_row, "labels", handle.get_stream().get());
      CUML_LOG_DEBUG("y_hat: %s", str.c_str());
      CUML_LOG_DEBUG("Score = %lf", score);
    }
  }

  void SetUp() override { basicTest(); }

 protected:
  DbscanInputs<T, IdxT> params;

  double score;
};

const std::vector<DbscanInputs<float, int>> inputsf2 = {
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL, ML::distance::DistanceType::Precomputed},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)13e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {20000,
   10000,
   10,
   0.01,
   2,
   2,
   (size_t)13e3,
   1234ULL,
   ML::distance::DistanceType::L2SqrtUnexpanded},
  {20000,
   100,
   5000,
   0.01,
   2,
   2,
   (size_t)13e3,
   1234ULL,
   ML::distance::DistanceType::L2SqrtUnexpanded}};

const std::vector<DbscanInputs<float, int64_t>> inputsf3 = {
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL, ML::distance::DistanceType::Precomputed},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)9e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {50000, 16, 5, 0.01, 2, 2, (size_t)9e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {20000,
   10000,
   10,
   0.01,
   2,
   2,
   (size_t)9e3,
   1234ULL,
   ML::distance::DistanceType::L2SqrtUnexpanded},
  {20000,
   100,
   5000,
   0.01,
   2,
   2,
   (size_t)9e3,
   1234ULL,
   ML::distance::DistanceType::L2SqrtUnexpanded}};

const std::vector<DbscanInputs<double, int>> inputsd2 = {
  {50000, 16, 5, 0.01, 2, 2, (size_t)13e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {10000, 16, 5, 0.01, 2, 2, (size_t)13e3, 1234ULL, ML::distance::DistanceType::Precomputed},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)13e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {100, 10000, 10, 0.01, 2, 2, (size_t)13e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {20000,
   10000,
   10,
   0.01,
   2,
   2,
   (size_t)13e3,
   1234ULL,
   ML::distance::DistanceType::L2SqrtUnexpanded},
  {20000,
   100,
   5000,
   0.01,
   2,
   2,
   (size_t)13e3,
   1234ULL,
   ML::distance::DistanceType::L2SqrtUnexpanded}};

const std::vector<DbscanInputs<double, int64_t>> inputsd3 = {
  {50000, 16, 5, 0.01, 2, 2, (size_t)9e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {10000, 16, 5, 0.01, 2, 2, (size_t)9e3, 1234ULL, ML::distance::DistanceType::Precomputed},
  {500, 16, 5, 0.01, 2, 2, (size_t)100, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {1000, 1000, 10, 0.01, 2, 2, (size_t)9e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {100, 10000, 10, 0.01, 2, 2, (size_t)9e3, 1234ULL, ML::distance::DistanceType::L2SqrtUnexpanded},
  {20000,
   10000,
   10,
   0.01,
   2,
   2,
   (size_t)9e3,
   1234ULL,
   ML::distance::DistanceType::L2SqrtUnexpanded},
  {20000,
   100,
   5000,
   0.01,
   2,
   2,
   (size_t)9e3,
   1234ULL,
   ML::distance::DistanceType::L2SqrtUnexpanded}};

typedef DbscanTest<float, int> DbscanTestF_Int;
TEST_P(DbscanTestF_Int, Result) { ASSERT_TRUE(score == 1.0); }

typedef DbscanTest<float, int64_t> DbscanTestF_Int64;
TEST_P(DbscanTestF_Int64, Result) { ASSERT_TRUE(score == 1.0); }

typedef DbscanTest<double, int> DbscanTestD_Int;
TEST_P(DbscanTestD_Int, Result) { ASSERT_TRUE(score == 1.0); }

typedef DbscanTest<double, int64_t> DbscanTestD_Int64;
TEST_P(DbscanTestD_Int64, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestF_Int, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestF_Int64, ::testing::ValuesIn(inputsf3));

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestD_Int, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestD_Int64, ::testing::ValuesIn(inputsd3));

template <typename T>
struct DBScan2DArrayInputs {
  const T* points;
  const int64_t* out;
  size_t n_row;
  // n_out allows to compare less labels than we have inputs
  // (some output labels can be ambiguous)
  size_t n_out;
  T eps;
  int min_pts;
  const int64_t* core_indices;  // Expected core_indices
  const T* sample_weight = nullptr;
};

template <typename T>
class Dbscan2DSimple : public ::testing::TestWithParam<DBScan2DArrayInputs<T>> {
 protected:
  void basicTest(Dbscan::EpsNnMethod eps_nn_method)
  {
    if (eps_nn_method == Dbscan::EpsNnMethod::RBC) { std::cout << "RBC test" << std::endl; }

    raft::handle_t handle;
    auto stream = handle.get_stream().get();

    params = ::testing::TestWithParam<DBScan2DArrayInputs<T>>::GetParam();

    rmm::device_uvector<T> inputs(params.n_row * 2, stream);
    rmm::device_uvector<int64_t> labels(params.n_row, stream);
    rmm::device_uvector<int64_t> labels_ref(params.n_out, stream);
    rmm::device_uvector<int64_t> core_sample_indices_d(params.n_row, stream);
    rmm::device_uvector<T> sample_weight_d(params.n_row, stream);

    raft::copy(inputs.data(), params.points, params.n_row * 2, stream);
    raft::copy(labels_ref.data(), params.out, params.n_out, stream);

    T* sample_weight = nullptr;
    if (params.sample_weight != nullptr) {
      raft::copy(sample_weight_d.data(), params.sample_weight, params.n_row, stream);
      sample_weight = sample_weight_d.data();
    }

    handle.sync_stream(stream);

    Dbscan::fit(handle,
                inputs.data(),
                (int)params.n_row,
                2,
                params.eps,
                params.min_pts,
                ML::distance::DistanceType::L2SqrtUnexpanded,
                labels.data(),
                core_sample_indices_d.data(),
                sample_weight,
                0,
                eps_nn_method);

    handle.sync_stream(handle.get_stream());

    score = adjusted_rand_index(handle, labels_ref.data(), labels.data(), (int)params.n_out);

    if (score < 1.0) {
      auto str = raft::arr2Str(labels_ref.data(), params.n_out, "labels_ref", stream);
      printf("y: %s\n", str.c_str());
      str = raft::arr2Str(labels.data(), params.n_row, "labels", stream);
      printf("y_hat: %s\n", str.c_str());
      printf("Score = %lf\n", score);
    }

    EXPECT_TRUE(MLCommon::devArrMatchHost(params.core_indices,
                                          core_sample_indices_d.data(),
                                          params.n_row,
                                          MLCommon::Compare<int64_t>(),
                                          stream));
  }

  void SetUp() override
  {
    basicTest(Dbscan::EpsNnMethod::BRUTE_FORCE);
    basicTest(Dbscan::EpsNnMethod::RBC);
  }

 protected:
  DBScan2DArrayInputs<T> params;

  double score;
};

// The input looks like a latin cross or a star with a chain:
//   .
// . . . . .
//   .
// There is 1 core-point (intersection of the bars)
// and the two points to the very right are not reachable from it
// So there should be one cluster (the plus/star on the left)
// and two noise points
const std::vector<float> test2d1_f = {0, 0, 1, 0, 1, 1, 1, -1, 2, 0, 3, 0, 4, 0};
const std::vector<double> test2d1_d(test2d1_f.begin(), test2d1_f.end());
const std::vector<int64_t> test2d1_l  = {0, 0, 0, 0, 0, -1, -1};
const std::vector<int64_t> test2d1c_l = {1, -1, -1, -1, -1, -1, -1};
// modified for weighted samples --> weights are shifted so that
// the rightmost point will be a core point as well
const std::vector<float> test2d1w_f = {1, 2, 1, 1, -1, 1, 3};
const std::vector<double> test2d1w_d(test2d1w_f.begin(), test2d1w_f.end());
const std::vector<int64_t> test2d1w_l  = {0, 0, 0, 0, 0, 1, 1};
const std::vector<int64_t> test2d1wc_l = {1, 6, -1, -1, -1, -1, -1};

// The input looks like a long two-barred (orhodox) cross or
// two stars next to each other:
//   .     .
// . . . . . .
//   .     .
// There are 2 core-points but they are not reachable from each other
// So there should be two clusters, both in the form of a plus/star
const std::vector<float> test2d2_f = {0, 0, 1, 0, 1, 1, 1, -1, 2, 0, 3, 0, 4, 0, 4, 1, 4, -1, 5, 0};
const std::vector<double> test2d2_d(test2d2_f.begin(), test2d2_f.end());
const std::vector<int64_t> test2d2_l  = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
const std::vector<int64_t> test2d2c_l = {1, 6, -1, -1, -1, -1, -1, -1, -1, -1};
// modified for weighted samples --> weight for the right center
// is negative that the whole right star is noise
const std::vector<float> test2d2w_f = {1, 1, 1, 1, 1, 1, -2, 1, 1, 1};
const std::vector<double> test2d2w_d(test2d2w_f.begin(), test2d2w_f.end());
const std::vector<int64_t> test2d2w_l  = {0, 0, 0, 0, 0, -1, -1, -1, -1, -1};
const std::vector<int64_t> test2d2wc_l = {1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

// The input looks like a two-barred (orhodox) cross or
// two stars sharing a link:
//   .   .
// . . . . .
//   .   .
// There are 2 core-points but they are not reachable from each other
// So there should be two clusters.
// However, the link that is shared between the stars
// actually has an ambiguous label (to the best of my knowledge)
// as it will depend on the order in which we process the core-points.
// Note that there are 9 input points, but only 8 labels for this reason
const std::vector<float> test2d3_f = {
  0,
  0,
  1,
  0,
  1,
  1,
  1,
  -1,
  3,
  0,
  3,
  1,
  3,
  -1,
  4,
  0,
  2,
  0,
};
const std::vector<double> test2d3_d(test2d3_f.begin(), test2d3_f.end());
const std::vector<int64_t> test2d3_l  = {0, 0, 0, 0, 1, 1, 1, 1};
const std::vector<int64_t> test2d3c_l = {1, 4, -1, -1, -1, -1, -1, -1, -1};

// ones for functional sample_weight testing
const std::vector<float> test2d_ones_f = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
const std::vector<double> test2d_ones_d(test2d_ones_f.begin(), test2d_ones_f.end());

const std::vector<DBScan2DArrayInputs<float>> inputs2d_f = {
  {test2d1_f.data(),
   test2d1_l.data(),
   test2d1_f.size() / 2,
   test2d1_l.size(),
   1.1f,
   4,
   test2d1c_l.data()},
  {test2d2_f.data(),
   test2d2_l.data(),
   test2d2_f.size() / 2,
   test2d2_l.size(),
   1.1f,
   4,
   test2d2c_l.data()},
  {test2d3_f.data(),
   test2d3_l.data(),
   test2d3_f.size() / 2,
   test2d3_l.size(),
   1.1f,
   4,
   test2d3c_l.data()},
  // add dummy sample weights
  {test2d1_f.data(),
   test2d1_l.data(),
   test2d1_f.size() / 2,
   test2d1_l.size(),
   1.1f,
   4,
   test2d1c_l.data(),
   test2d_ones_f.data()},
  {test2d2_f.data(),
   test2d2_l.data(),
   test2d2_f.size() / 2,
   test2d2_l.size(),
   1.1f,
   4,
   test2d2c_l.data(),
   test2d_ones_f.data()},
  {test2d3_f.data(),
   test2d3_l.data(),
   test2d3_f.size() / 2,
   test2d3_l.size(),
   1.1f,
   4,
   test2d3c_l.data(),
   test2d_ones_f.data()},
  // special sample_weight cases
  {test2d1_f.data(),
   test2d1w_l.data(),
   test2d1_f.size() / 2,
   test2d1w_l.size(),
   1.1f,
   4,
   test2d2wc_l.data(),
   test2d2w_f.data()},
  {test2d2_f.data(),
   test2d2w_l.data(),
   test2d2_f.size() / 2,
   test2d2w_l.size(),
   1.1f,
   4,
   test2d2wc_l.data(),
   test2d2w_f.data()},
};

const std::vector<DBScan2DArrayInputs<double>> inputs2d_d = {
  {test2d1_d.data(),
   test2d1_l.data(),
   test2d1_d.size() / 2,
   test2d1_l.size(),
   1.1,
   4,
   test2d1c_l.data()},
  {test2d2_d.data(),
   test2d2_l.data(),
   test2d2_d.size() / 2,
   test2d2_l.size(),
   1.1,
   4,
   test2d2c_l.data()},
  {test2d3_d.data(),
   test2d3_l.data(),
   test2d3_d.size() / 2,
   test2d3_l.size(),
   1.1,
   4,
   test2d3c_l.data()},
  // add dummy sample weights
  {test2d1_d.data(),
   test2d1_l.data(),
   test2d1_d.size() / 2,
   test2d1_l.size(),
   1.1,
   4,
   test2d1c_l.data(),
   test2d_ones_d.data()},
  {test2d2_d.data(),
   test2d2_l.data(),
   test2d2_d.size() / 2,
   test2d2_l.size(),
   1.1,
   4,
   test2d2c_l.data(),
   test2d_ones_d.data()},
  {test2d3_d.data(),
   test2d3_l.data(),
   test2d3_d.size() / 2,
   test2d3_l.size(),
   1.1,
   4,
   test2d3c_l.data(),
   test2d_ones_d.data()},
  // special sample_weight cases
  {test2d1_d.data(),
   test2d1w_l.data(),
   test2d1_d.size() / 2,
   test2d1w_l.size(),
   1.1f,
   4,
   test2d1wc_l.data(),
   test2d1w_d.data()},
  {test2d2_d.data(),
   test2d2w_l.data(),
   test2d2_d.size() / 2,
   test2d2w_l.size(),
   1.1f,
   4,
   test2d2wc_l.data(),
   test2d2w_d.data()},
};

typedef Dbscan2DSimple<float> Dbscan2DSimple_F;
TEST_P(Dbscan2DSimple_F, Result) { ASSERT_TRUE(score == 1.0); }

typedef Dbscan2DSimple<double> Dbscan2DSimple_D;
TEST_P(Dbscan2DSimple_D, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(DbscanTests, Dbscan2DSimple_F, ::testing::ValuesIn(inputs2d_f));

INSTANTIATE_TEST_CASE_P(DbscanTests, Dbscan2DSimple_D, ::testing::ValuesIn(inputs2d_d));

}  // end namespace ML
