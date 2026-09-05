/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.h"

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <timeSeries/fillna.cuh>

#include <random>
#include <vector>

namespace MLCommon {
namespace TimeSeries {

using namespace std;

struct SeriesDescriptor {
  int leading_nan;
  int random_nan;
  int trailing_nan;
};

template <typename T>
struct FillnaInputs {
  int batch_size;
  int n_obs;
  std::vector<SeriesDescriptor> descriptors;
  unsigned long long int seed;
  T tolerance;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const FillnaInputs<T>& dims)
{
  return os;
}

template <typename T>
class FillnaTest : public ::testing::TestWithParam<FillnaInputs<T>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<FillnaInputs<T>>::GetParam();

    rmm::device_uvector<T> y(params.n_obs * params.batch_size, handle.get_stream());

    std::vector<T> h_y(params.n_obs * params.batch_size);

    /* Generate random data */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> real_distribution(-2.0, 2.0);
    std::uniform_int_distribution<int> int_distribution(0, params.n_obs - 1);
    for (int i = 0; i < params.n_obs * params.batch_size; i++)
      h_y[i] = real_distribution(generator);
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.descriptors[bid].leading_nan; i++)
        h_y[bid * params.n_obs + i] = nan("");
      for (int i = 0; i < params.descriptors[bid].trailing_nan; i++)
        h_y[(bid + 1) * params.n_obs - 1 - i] = nan("");
      for (int i = 0; i < params.descriptors[bid].random_nan; i++) {
        h_y[bid * params.n_obs + int_distribution(generator)] = nan("");
      }
    }

    /* Copy to device */
    raft::update_device(
      y.data(), h_y.data(), params.n_obs * params.batch_size, handle.get_stream());
    handle.sync_stream(handle.get_stream());

    /* Compute using tested prims */
    fillna(y.data(), params.batch_size, params.n_obs, handle.get_stream().get());

    /* Compute reference results.
     * Note: this is done with a sliding window: we find ranges of missing
     * values bordered by valid values at indices `start` and `end`.
     * Special cases on extremities are also handled with the special values
     * -1 for `start` and `n_obs` for `end`.
     */
    for (int bid = 0; bid < params.batch_size; bid++) {
      int start = -1;
      int end   = 0;
      while (start < params.n_obs - 1) {
        if (!std::isnan(h_y[bid * params.n_obs + start + 1])) {
          start++;
          end = start + 1;
        } else if (end < params.n_obs && std::isnan(h_y[bid * params.n_obs + end])) {
          end++;
        } else {
          if (start == -1) {
            T value = h_y[bid * params.n_obs + end];
            for (int j = 0; j < end; j++) {
              h_y[bid * params.n_obs + j] = value;
            }
          } else if (end == params.n_obs) {
            T value = h_y[bid * params.n_obs + start];
            for (int j = start + 1; j < params.n_obs; j++) {
              h_y[bid * params.n_obs + j] = value;
            }
          } else {
            T value0 = h_y[bid * params.n_obs + start];
            T value1 = h_y[bid * params.n_obs + end];
            for (int j = start + 1; j < end; j++) {
              T coef                      = (T)(j - start) / (T)(end - start);
              h_y[bid * params.n_obs + j] = ((T)1 - coef) * value0 + coef * value1;
            }
          }
          start = end;
          end++;
        }
      }
    }

    /* Check results */
    match = devArrMatchHost(h_y.data(),
                            y.data(),
                            params.n_obs * params.batch_size,
                            MLCommon::CompareApprox<T>(params.tolerance),
                            handle.get_stream().get());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  FillnaInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<FillnaInputs<float>> inputsf = {
  {1, 20, {{1, 5, 1}}, 12345U, 1e-6},
  {3, 42, {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}, 12345U, 1e-6},
  {4, 100, {{70, 0, 0}, {0, 20, 0}, {0, 0, 63}, {31, 25, 33}, {20, 15, 42}}, 12345U, 1e-6},
};

const std::vector<FillnaInputs<double>> inputsd = {
  {1, 20, {{1, 5, 1}}, 12345U, 1e-6},
  {3, 42, {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}, 12345U, 1e-6},
  {4, 100, {{70, 0, 0}, {0, 20, 0}, {0, 0, 63}, {31, 25, 33}, {20, 15, 42}}, 12345U, 1e-6},
};

typedef FillnaTest<float> FillnaTestF;
TEST_P(FillnaTestF, Result) { EXPECT_TRUE(match); }

typedef FillnaTest<double> FillnaTestD;
TEST_P(FillnaTestD, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(FillnaTests, FillnaTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(FillnaTests, FillnaTestD, ::testing::ValuesIn(inputsd));

}  // namespace TimeSeries
}  // namespace MLCommon
