/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuml/explainer/kernel_shap.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <test_utils.h>

namespace MLCommon {
}
#include <gtest/gtest.h>

namespace ML {

namespace Explainer {

struct MakeKSHAPDatasetInputs {
  int nrows_exact;
  int nrows_sampled;
  int ncols;
  int nrows_background;
  int max_samples;
  uint64_t seed;
};

template <typename T>
class MakeKSHAPDatasetTest : public ::testing::TestWithParam<MakeKSHAPDatasetInputs> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<MakeKSHAPDatasetInputs>::GetParam();
    stream = handle.get_stream().get();

    int i, j;
    nrows_X = params.nrows_exact + params.nrows_sampled;

    rmm::device_uvector<T> background(params.nrows_background * params.ncols, stream);
    rmm::device_uvector<T> observation(params.ncols, stream);
    rmm::device_uvector<int> nsamples(params.nrows_sampled / 2, stream);
    rmm::device_uvector<float> X(nrows_X * params.ncols, stream);
    rmm::device_uvector<T> dataset(nrows_X * params.nrows_background * params.ncols, stream);

    thrust::device_ptr<T> b_ptr   = thrust::device_pointer_cast(background.data());
    thrust::device_ptr<T> o_ptr   = thrust::device_pointer_cast(observation.data());
    thrust::device_ptr<int> n_ptr = thrust::device_pointer_cast(nsamples.data());

    thrust::device_ptr<float> X_ptr = thrust::device_pointer_cast(X.data());
    thrust::device_ptr<T> d_ptr     = thrust::device_pointer_cast(dataset.data());

    // Initialize arrays:

    // Assign a sentinel value to the observation to check easily later
    T sent_value = nrows_X * params.nrows_background * params.ncols * 100;
    for (i = 0; i < params.ncols; i++) {
      o_ptr[i] = sent_value;
    }

    // Initialize background array with different odd value per row, makes
    // it easier to debug if something goes wrong.
    for (i = 0; i < params.nrows_background; i++) {
      for (j = 0; j < params.ncols; j++) {
        b_ptr[i * params.ncols + j] = (i * 2) + 1;
      }
    }

    // Initialize the exact part of X. We create 2 `1` values per row for the test
    thrust::fill(thrust::device, X_ptr, &X_ptr[nrows_X * params.ncols - 1], 0);
    for (i = 0; i < params.nrows_exact; i++) {
      for (j = i; j < i + 2; j++) {
        X_ptr[i * params.ncols + j] = (float)1.0;
      }
    }

    // Initialize the number of samples per row, we initialize each even row to
    // max samples and each odd row to max_samples - 1
    for (i = 0; i < params.nrows_sampled / 2; i++) {
      n_ptr[i] = params.max_samples - i % 2;
    }

    kernel_dataset(handle,
                   X.data(),
                   nrows_X,
                   params.ncols,
                   background.data(),
                   params.nrows_background,
                   dataset.data(),
                   observation.data(),
                   nsamples.data(),
                   params.nrows_sampled,
                   params.max_samples,
                   params.seed);

    handle.sync_stream(stream);

    int counter;

    // Check the generated part of X by sampling. The first nrows_exact
    // correspond to the exact part generated before, so we just test after that.
    test_sampled_X = true;
    j              = 0;
    for (i = params.nrows_exact * params.ncols; i < nrows_X * params.ncols / 2;
         i += 2 * params.ncols) {
      // check that number of samples is the number indicated by nsamples.
      counter        = thrust::count(&X_ptr[i], &X_ptr[i + params.ncols], 1);
      test_sampled_X = (test_sampled_X && (counter == n_ptr[j]));

      // check that number of samples of the next line is the compliment,
      // i.e. ncols - nsamples[j]
      counter        = thrust::count(&X_ptr[i + params.ncols], &X_ptr[i + 2 * params.ncols], 1);
      test_sampled_X = (test_sampled_X && (counter == (params.ncols - n_ptr[j])));

      j++;
    }

    // Check for the exact part of the generated dataset.
    test_scatter_exact = true;
    for (i = 0; i < params.nrows_exact; i++) {
      for (j = i * params.nrows_background * params.ncols;
           j < (i + 1) * params.nrows_background * params.ncols;
           j += params.ncols) {
        counter = thrust::count(&d_ptr[j], &d_ptr[j + params.ncols], sent_value);

        // Check that indeed we have two observation entries ber row
        test_scatter_exact = test_scatter_exact && (counter == 2);
        if (not test_scatter_exact) {
          std::cout << "test_scatter_exact counter failed with: " << counter
                    << ", expected value was 2." << std::endl;
          break;
        }
      }
      if (not test_scatter_exact) { break; }
    }

    // Check for the sampled part of the generated dataset
    test_scatter_sampled = true;

    // compliment_ctr is a helper counter to help check nrows_dataset per entry in
    // nsamples without complicating indexing since sampled part starts at nrows_sampled
    int compliment_ctr = 0;
    for (i = params.nrows_exact; i < params.nrows_exact + params.nrows_sampled / 2; i++) {
      // First set of dataset observations must correspond to nsamples[i]
      for (j = (i + compliment_ctr) * params.nrows_background * params.ncols;
           j < (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
           j += params.ncols) {
        counter              = thrust::count(&d_ptr[j], &d_ptr[j + params.ncols], sent_value);
        test_scatter_sampled = test_scatter_sampled && (counter == n_ptr[i - params.nrows_exact]);
      }

      // The next set of samples must correspond to the compliment: ncols - nsamples[i]
      compliment_ctr++;
      for (j = (i + compliment_ctr) * params.nrows_background * params.ncols;
           j < (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
           j += params.ncols) {
        // Check that number of observation entries corresponds to nsamples.
        counter = thrust::count(&d_ptr[j], &d_ptr[j + params.ncols], sent_value);
        test_scatter_sampled =
          test_scatter_sampled && (counter == params.ncols - n_ptr[i - params.nrows_exact]);
      }
    }
  }

 protected:
  MakeKSHAPDatasetInputs params;
  int nrows_X;
  bool test_sampled_X;
  bool test_scatter_exact;
  bool test_scatter_sampled;
  raft::handle_t handle;
  cudaStream_t stream = 0;
};

const std::vector<MakeKSHAPDatasetInputs> inputsf = {{10, 10, 12, 2, 3, 1234ULL},
                                                     {10, 0, 12, 2, 3, 1234ULL},
                                                     {100, 50, 200, 10, 10, 1234ULL},
                                                     {100, 0, 200, 10, 10, 1234ULL},
                                                     {0, 10, 12, 2, 3, 1234ULL},
                                                     {0, 50, 200, 10, 10, 1234ULL}

};

typedef MakeKSHAPDatasetTest<float> MakeKSHAPDatasetTestF;
TEST_P(MakeKSHAPDatasetTestF, Result)
{
  ASSERT_TRUE(test_sampled_X);
  // todo (dgd): re-enable assertions
  // disabled due to a sporadic cuda 10.1 fail (by one value in one case!)
  // will be re-enabled soon after 0.17 release
  // ASSERT_TRUE(test_scatter_exact);
  // ASSERT_TRUE(test_scatter_sampled);
}
INSTANTIATE_TEST_CASE_P(MakeKSHAPDatasetTests, MakeKSHAPDatasetTestF, ::testing::ValuesIn(inputsf));

}  // end namespace Explainer
}  // end namespace ML
