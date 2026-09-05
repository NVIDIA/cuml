/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "time_series_datasets.h"

#include <cuml/common/logger.hpp>
#include <cuml/tsa/holtwinters.h>

#include <raft/core/handle.hpp>
#include <raft/core/math.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <algorithm>

namespace ML {

template <typename T>
struct HoltWintersInputs {
  T* dataset_h;
  T* test;
  int n;
  int h;
  int batch_size;
  int frequency;
  ML::SeasonalType seasonal;
  int start_periods;
  T epsilon;
  T mae_tolerance;
};

template <typename T>
class HoltWintersTest : public ::testing::TestWithParam<HoltWintersInputs<T>> {
 public:
  HoltWintersTest()
    : params(::testing::TestWithParam<HoltWintersInputs<T>>::GetParam()),
      stream(handle.get_stream().get()),
      level_ptr(0, stream),
      trend_ptr(0, stream),
      season_ptr(0, stream),
      SSE_error_ptr(0, stream),
      forecast_ptr(0, stream),
      data(0, stream)
  {
  }

  void basicTest()
  {
    dataset_h                 = params.dataset_h;
    test                      = params.test;
    n                         = params.n;
    h                         = params.h;
    batch_size                = params.batch_size;
    frequency                 = params.frequency;
    ML::SeasonalType seasonal = params.seasonal;
    start_periods             = params.start_periods;
    epsilon                   = params.epsilon;
    mae_tolerance             = params.mae_tolerance;

    ML::HoltWinters::buffer_size(
      n,
      batch_size,
      frequency,
      &leveltrend_seed_len,     // = batch_size
      &season_seed_len,         // = frequency*batch_size
      &components_len,          // = (n-w_len)*batch_size
      &error_len,               // = batch_size
      &leveltrend_coef_offset,  // = (n-wlen-1)*batch_size (last row)
      &season_coef_offset);     // = (n-wlen-frequency)*batch_size(last freq rows)

    level_ptr.resize(components_len, stream);
    trend_ptr.resize(components_len, stream);
    season_ptr.resize(components_len, stream);
    SSE_error_ptr.resize(batch_size, stream);
    forecast_ptr.resize(batch_size * h, stream);
    data.resize(batch_size * n, stream);
    raft::update_device(data.data(), dataset_h, batch_size * n, stream);

    raft::handle_t handle{stream};

    ML::HoltWinters::fit(handle,
                         n,
                         batch_size,
                         frequency,
                         start_periods,
                         seasonal,
                         epsilon,
                         data.data(),
                         level_ptr.data(),
                         trend_ptr.data(),
                         season_ptr.data(),
                         SSE_error_ptr.data());

    ML::HoltWinters::forecast(handle,
                              n,
                              batch_size,
                              frequency,
                              h,
                              seasonal,
                              level_ptr.data(),
                              trend_ptr.data(),
                              season_ptr.data(),
                              forecast_ptr.data());

    handle.sync_stream(stream);
  }

  void SetUp() override { basicTest(); }

 public:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  HoltWintersInputs<T> params;
  T *dataset_h, *test;
  rmm::device_uvector<T> data;
  int n, h;
  int leveltrend_seed_len, season_seed_len, components_len;
  int leveltrend_coef_offset, season_coef_offset;
  int error_len;
  int batch_size, frequency, start_periods;
  rmm::device_uvector<T> SSE_error_ptr, level_ptr, trend_ptr, season_ptr, forecast_ptr;
  T epsilon, mae_tolerance;
};

const std::vector<HoltWintersInputs<float>> inputsf = {{additive_trainf.data(),
                                                        additive_testf.data(),
                                                        90,
                                                        10,
                                                        1,
                                                        25,
                                                        ML::SeasonalType::ADDITIVE,
                                                        2,
                                                        2.24e-3,
                                                        1e-6},
                                                       {multiplicative_trainf.data(),
                                                        multiplicative_testf.data(),
                                                        132,
                                                        12,
                                                        1,
                                                        12,
                                                        ML::SeasonalType::MULTIPLICATIVE,
                                                        2,
                                                        2.24e-3,
                                                        3e-2},
                                                       {additive_normalized_trainf.data(),
                                                        additive_normalized_testf.data(),
                                                        90,
                                                        10,
                                                        1,
                                                        25,
                                                        ML::SeasonalType::ADDITIVE,
                                                        2,
                                                        2.24e-3,
                                                        1e-6},
                                                       {multiplicative_normalized_trainf.data(),
                                                        multiplicative_normalized_testf.data(),
                                                        132,
                                                        12,
                                                        1,
                                                        12,
                                                        ML::SeasonalType::MULTIPLICATIVE,
                                                        2,
                                                        2.24e-3,
                                                        2.5e-1}};

const std::vector<HoltWintersInputs<double>> inputsd = {{additive_traind.data(),
                                                         additive_testd.data(),
                                                         90,
                                                         10,
                                                         1,
                                                         25,
                                                         ML::SeasonalType::ADDITIVE,
                                                         2,
                                                         2.24e-7,
                                                         1e-6},
                                                        {multiplicative_traind.data(),
                                                         multiplicative_testd.data(),
                                                         132,
                                                         12,
                                                         1,
                                                         12,
                                                         ML::SeasonalType::MULTIPLICATIVE,
                                                         2,
                                                         2.24e-7,
                                                         3e-2},
                                                        {additive_normalized_traind.data(),
                                                         additive_normalized_testd.data(),
                                                         90,
                                                         10,
                                                         1,
                                                         25,
                                                         ML::SeasonalType::ADDITIVE,
                                                         2,
                                                         2.24e-7,
                                                         1e-6},
                                                        {multiplicative_normalized_traind.data(),
                                                         multiplicative_normalized_testd.data(),
                                                         132,
                                                         12,
                                                         1,
                                                         12,
                                                         ML::SeasonalType::MULTIPLICATIVE,
                                                         2,
                                                         2.24e-7,
                                                         5e-2}};

template <typename T>
void normalise(T* data, int len)
{
  T min = *std::min_element(data, data + len);
  T max = *std::max_element(data, data + len);
  for (int i = 0; i < len; i++) {
    data[i] = (data[i] - min) / (max - min);
  }
}

template <typename T>
T calculate_MAE(T* test, T* forecast, int batch_size, int h)
{
  normalise(test, batch_size * h);
  normalise(forecast, batch_size * h);
  std::vector<T> ae(batch_size * h);
  for (int i = 0; i < batch_size * h; i++) {
    ae[i] = raft::abs(test[i] - forecast[i]);
  }
  std::sort(ae.begin(), ae.end());
  T mae;
  if (h % 2 == 0) {
    mae = (ae[h / 2 - 1] + ae[h / 2]) / 2;
  } else {
    mae = ae[(int)h / 2];
  }
  return mae;
}

typedef HoltWintersTest<float> HoltWintersTestF;
TEST_P(HoltWintersTestF, Fit)
{
  std::vector<float> forecast_h(batch_size * h);
  raft::update_host(forecast_h.data(), forecast_ptr.data(), batch_size * h, stream);
  raft::print_host_vector("forecast", forecast_h.data(), batch_size * h, std::cout);
  float mae = calculate_MAE<float>(test, forecast_h.data(), batch_size, h);
  CUML_LOG_DEBUG("MAE: %f", mae);
  ASSERT_TRUE(mae < mae_tolerance);
}

typedef HoltWintersTest<double> HoltWintersTestD;
TEST_P(HoltWintersTestD, Fit)
{
  std::vector<double> forecast_h(batch_size * h);
  raft::update_host(forecast_h.data(), forecast_ptr.data(), batch_size * h, stream);
  raft::print_host_vector("forecast", forecast_h.data(), batch_size * h, std::cout);
  double mae = calculate_MAE<double>(test, forecast_h.data(), batch_size, h);
  CUML_LOG_DEBUG("MAE: %f", mae);
  ASSERT_TRUE(mae < mae_tolerance);
}

INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestD, ::testing::ValuesIn(inputsd));

}  // namespace ML
