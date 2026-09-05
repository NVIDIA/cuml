/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../prims/test_utils.h"
#include "hdbscan_inputs.hpp"

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/distance_type.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/op/sort.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/tuple>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cuvs/cluster/agglomerative.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>
#include <gtest/gtest.h>
#include <hdbscan/detail/condense.cuh>
#include <hdbscan/detail/extract.cuh>
#include <hdbscan/detail/reachability.cuh>

#include <vector>

namespace ML {
namespace HDBSCAN {

using namespace std;

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const HDBSCANInputs<T, IdxT>& dims)
{
  return os;
}

template <typename T, typename IdxT>
class HDBSCANTest : public ::testing::TestWithParam<HDBSCANInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<HDBSCANInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<T> data(params.n_row * params.n_col, handle.get_stream());

    // Allocate result labels and expected labels on device
    rmm::device_uvector<IdxT> labels_ref(params.n_row, handle.get_stream());

    raft::copy(data.data(), params.data.data(), data.size(), handle.get_stream());
    raft::copy(labels_ref.data(), params.expected_labels.data(), params.n_row, handle.get_stream());

    rmm::device_uvector<IdxT> out_children(params.n_row * 2, handle.get_stream());
    rmm::device_uvector<T> out_deltas(params.n_row, handle.get_stream());

    rmm::device_uvector<IdxT> out_sizes(params.n_row * 2, handle.get_stream());

    rmm::device_uvector<IdxT> out_labels(params.n_row, handle.get_stream());

    rmm::device_uvector<IdxT> mst_src(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<IdxT> mst_dst(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<T> mst_weights(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<T> core_dists(params.n_row, handle.get_stream());

    rmm::device_uvector<T> out_probabilities(params.n_row, handle.get_stream());

    default_logger().set_level(rapids_logger::level_enum::debug);

    HDBSCAN::Common::hdbscan_output<IdxT, T> out(handle,
                                                 params.n_row,
                                                 out_labels.data(),
                                                 out_probabilities.data(),
                                                 out_children.data(),
                                                 out_sizes.data(),
                                                 out_deltas.data(),
                                                 mst_src.data(),
                                                 mst_dst.data(),
                                                 mst_weights.data());

    HDBSCAN::Common::HDBSCANParams hdbscan_params;
    hdbscan_params.min_cluster_size = params.min_cluster_size;
    hdbscan_params.min_samples      = params.min_pts;

    hdbscan(handle,
            data.data(),
            params.n_row,
            params.n_col,
            ML::distance::DistanceType::L2SqrtExpanded,
            hdbscan_params,
            out,
            core_dists.data());

    handle.sync_stream(handle.get_stream().get());

    score = raft::stats::adjusted_rand_index(
      out.get_labels(), labels_ref.data(), params.n_row, handle.get_stream().get());

    if (score < 0.85) {
      std::cout << "Test failed. score=" << score << std::endl;
      raft::print_device_vector("actual labels", out.get_labels(), params.n_row, std::cout);
      raft::print_device_vector("expected labels", labels_ref.data(), params.n_row, std::cout);
    }
  }

  void SetUp() override { basicTest(); }

 protected:
  HDBSCANInputs<T, IdxT> params;
  IdxT* labels_ref;

  double score;
};

typedef HDBSCANTest<float, int64_t> HDBSCANTestF_Int;
TEST_P(HDBSCANTestF_Int, Result) { EXPECT_TRUE(score >= 0.85); }

INSTANTIATE_TEST_CASE_P(HDBSCANTest, HDBSCANTestF_Int, ::testing::ValuesIn(hdbscan_inputsf2));

template <typename T, typename IdxT>
class ClusterCondensingTest : public ::testing::TestWithParam<ClusterCondensingInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<ClusterCondensingInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<IdxT> mst_src(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<IdxT> mst_dst(params.n_row - 1, handle.get_stream());
    rmm::device_uvector<T> mst_data(params.n_row - 1, handle.get_stream());

    raft::copy(mst_src.data(), params.mst_src.data(), params.mst_src.size(), handle.get_stream());

    raft::copy(mst_dst.data(), params.mst_dst.data(), params.mst_dst.size(), handle.get_stream());

    raft::copy(
      mst_data.data(), params.mst_data.data(), params.mst_data.size(), handle.get_stream());

    rmm::device_uvector<IdxT> expected_device(params.expected.size(), handle.get_stream());
    raft::copy(
      expected_device.data(), params.expected.data(), params.expected.size(), handle.get_stream());

    rmm::device_uvector<IdxT> out_children(params.n_row * 2, handle.get_stream());

    rmm::device_uvector<IdxT> out_size(params.n_row, handle.get_stream());

    rmm::device_uvector<T> out_delta(params.n_row, handle.get_stream());

    default_logger().set_level(rapids_logger::level_enum::debug);

    raft::sparse::op::coo_sort_by_weight(
      mst_src.data(), mst_dst.data(), mst_data.data(), (IdxT)mst_src.size(), handle.get_stream());

    /**
     * Build dendrogram of MST
     */
    cuvs::cluster::agglomerative::helpers::build_dendrogram(
      handle,
      raft::make_device_vector_view<const IdxT, IdxT>(mst_src.data(), params.n_row - 1),
      raft::make_device_vector_view<const IdxT, IdxT>(mst_dst.data(), params.n_row - 1),
      raft::make_device_vector_view<const T, IdxT>(mst_data.data(), params.n_row - 1),
      raft::make_device_matrix_view<IdxT, IdxT, raft::row_major>(
        out_children.data(), params.n_row - 1, 2),
      raft::make_device_vector_view<T, IdxT>(out_delta.data(), params.n_row - 1),
      raft::make_device_vector_view<IdxT, IdxT>(out_size.data(), params.n_row - 1));

    /**
     * Condense Hierarchy
     */
    HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle, params.n_row);
    HDBSCAN::detail::Condense::build_condensed_hierarchy(handle,
                                                         out_children.data(),
                                                         out_delta.data(),
                                                         out_size.data(),
                                                         params.min_cluster_size,
                                                         params.n_row,
                                                         condensed_tree);

    handle.sync_stream(handle.get_stream());

    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());
    rmm::device_uvector<T> stabilities(condensed_tree.get_n_clusters(), handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> inverse_label_map(0, handle.get_stream());

    HDBSCAN::detail::Extract::extract_clusters(handle,
                                               condensed_tree,
                                               params.n_row,
                                               labels.data(),
                                               stabilities.data(),
                                               probabilities.data(),
                                               label_map.data(),
                                               HDBSCAN::Common::CLUSTER_SELECTION_METHOD::EOM,
                                               inverse_label_map,
                                               false);

    //    CUML_LOG_DEBUG("Evaluating results");
    //    if (params.expected.size() == params.n_row) {
    //      score = MLCommon::Metrics::compute_adjusted_rand_index(
    //        labels.data(), expected_device.data(), params.n_row,
    //        handle.get_stream());
    //    } else {
    //      score = 1.0;
    //    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  ClusterCondensingInputs<T, IdxT> params;

  double score;
};

#if 0
// gtest-1.11.0 makes it a runtime error to define and not instantiate this test case.

typedef ClusterCondensingTest<float, int64_t> ClusterCondensingTestF_Int;
TEST_P(ClusterCondensingTestF_Int, Result) { EXPECT_TRUE(score == 1.0); }

// This will be reactivate in 21.08 with better, contrived examples to
// test Cluster Condensation correctly
// INSTANTIATE_TEST_CASE_P(ClusterCondensingTest, ClusterCondensingTestF_Int,
//                         ::testing::ValuesIn(cluster_condensing_inputs));

#endif

template <typename T, typename IdxT>
class ClusterSelectionTest : public ::testing::TestWithParam<ClusterSelectionInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<ClusterSelectionInputs<T, IdxT>>::GetParam();

    default_logger().set_level(rapids_logger::level_enum::debug);

    rmm::device_uvector<IdxT> condensed_parents(params.condensed_parents.size(),
                                                handle.get_stream());
    rmm::device_uvector<IdxT> condensed_children(params.condensed_children.size(),
                                                 handle.get_stream());
    rmm::device_uvector<T> condensed_lambdas(params.condensed_lambdas.size(), handle.get_stream());
    rmm::device_uvector<IdxT> condensed_sizes(params.condensed_sizes.size(), handle.get_stream());

    // outputs
    rmm::device_uvector<T> stabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());

    raft::copy(condensed_parents.data(),
               params.condensed_parents.data(),
               condensed_parents.size(),
               handle.get_stream());

    raft::copy(condensed_children.data(),
               params.condensed_children.data(),
               condensed_children.size(),
               handle.get_stream());

    raft::copy(condensed_lambdas.data(),
               params.condensed_lambdas.data(),
               condensed_lambdas.size(),
               handle.get_stream());

    raft::copy(condensed_sizes.data(),
               params.condensed_sizes.data(),
               condensed_sizes.size(),
               handle.get_stream());

    ML::HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle,
                                                                    params.n_row,
                                                                    params.condensed_parents.size(),
                                                                    condensed_parents.data(),
                                                                    condensed_children.data(),
                                                                    condensed_lambdas.data(),
                                                                    condensed_sizes.data());

    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> inverse_label_map(0, handle.get_stream());

    ML::HDBSCAN::detail::Extract::extract_clusters(handle,
                                                   condensed_tree,
                                                   params.n_row,
                                                   labels.data(),
                                                   stabilities.data(),
                                                   probabilities.data(),
                                                   label_map.data(),
                                                   params.cluster_selection_method,
                                                   inverse_label_map,
                                                   params.allow_single_cluster,
                                                   static_cast<IdxT>(0),
                                                   params.cluster_selection_epsilon);

    handle.sync_stream(handle.get_stream().get());

    ASSERT_TRUE(MLCommon::devArrMatch(probabilities.data(),
                                      params.probabilities.data(),
                                      params.n_row,
                                      MLCommon::CompareApprox<float>(1e-4),
                                      handle.get_stream().get()));

    rmm::device_uvector<IdxT> labels_ref(params.n_row, handle.get_stream().get());
    raft::update_device(labels_ref.data(), params.labels.data(), params.n_row, handle.get_stream().get());
    score = raft::stats::adjusted_rand_index(
      labels.data(), labels_ref.data(), params.n_row, handle.get_stream().get());
    handle.sync_stream(handle.get_stream().get());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  ClusterSelectionInputs<T, IdxT> params;
  T score;
};

typedef ClusterSelectionTest<float, int64_t> ClusterSelectionTestF_Int;
TEST_P(ClusterSelectionTestF_Int, Result) { EXPECT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(ClusterSelectionTest,
                        ClusterSelectionTestF_Int,
                        ::testing::ValuesIn(cluster_selection_inputs));

template <typename IdxT>
void transformLabels(const raft::handle_t& handle, IdxT* labels, IdxT* label_map, IdxT m)
{
  thrust::transform(
    handle.get_thrust_policy(), labels, labels + m, labels, [label_map] __device__(IdxT label) {
      if (label != -1) return label_map[label];
      return static_cast<IdxT>(-1);
    });
}

// This test was constructed in the following manner: The same condensed tree and set of selected
// clusters need to be passed to the reference implementation and then compare the results from
// cuML and the reference implementation for an approximate match of probabilities. To fetch the
// condensed hierarchy in the same format as required by the reference implementation, a simple
// python script can be written:
// 1. Print the parents, children, lambdas and sizes array of the condensed hierarchy.
// 2. Convert them into a list ``condensed_tree`` of tuples where each tuples is of the form.
//    ``(parents[i], children[i], lambdas[i], sizes[i])``
// 3. Convert the list into a numpy array with the following command:
//    ``condensed_tree_array = np.array(condened_tree, dtype=[('parent', np.intp), ('child',
//                                      np.intp), ('lambda_val', float), ('child_size',
//                                      np.intp)])``
// 4. Store it in a pickle file.
// The reference source code is modified in the following way: Edit the raw tree in the init
// function of the PredictionData object in prediction.py by loading it from the pickle file. Also
// edit the selected clusters array. Do the same in the all_points_membership_vectors function and
// the approximate_predict functions.
template <typename T, typename IdxT>
class AllPointsMembershipVectorsTest
  : public ::testing::TestWithParam<AllPointsMembershipVectorsInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<AllPointsMembershipVectorsInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<IdxT> condensed_parents(params.condensed_parents.size(),
                                                handle.get_stream());
    rmm::device_uvector<IdxT> condensed_children(params.condensed_children.size(),
                                                 handle.get_stream());
    rmm::device_uvector<T> condensed_lambdas(params.condensed_lambdas.size(), handle.get_stream());
    rmm::device_uvector<IdxT> condensed_sizes(params.condensed_sizes.size(), handle.get_stream());

    raft::copy(condensed_parents.data(),
               params.condensed_parents.data(),
               condensed_parents.size(),
               handle.get_stream());

    raft::copy(condensed_children.data(),
               params.condensed_children.data(),
               condensed_children.size(),
               handle.get_stream());

    raft::copy(condensed_lambdas.data(),
               params.condensed_lambdas.data(),
               condensed_lambdas.size(),
               handle.get_stream());

    raft::copy(condensed_sizes.data(),
               params.condensed_sizes.data(),
               condensed_sizes.size(),
               handle.get_stream());

    rmm::device_uvector<T> data(params.n_row * params.n_col, handle.get_stream());
    raft::copy(data.data(), params.data.data(), data.size(), handle.get_stream());

    ML::HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle,
                                                                    params.n_row,
                                                                    params.condensed_parents.size(),
                                                                    condensed_parents.data(),
                                                                    condensed_children.data(),
                                                                    condensed_lambdas.data(),
                                                                    condensed_sizes.data());

    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());

    // intermediate outputs
    rmm::device_uvector<T> stabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> inverse_label_map(0, handle.get_stream());

    int n_selected_clusters =
      ML::HDBSCAN::detail::Extract::extract_clusters(handle,
                                                     condensed_tree,
                                                     params.n_row,
                                                     labels.data(),
                                                     stabilities.data(),
                                                     probabilities.data(),
                                                     label_map.data(),
                                                     params.cluster_selection_method,
                                                     inverse_label_map,
                                                     params.allow_single_cluster,
                                                     static_cast<IdxT>(0),
                                                     params.cluster_selection_epsilon);

    rmm::device_uvector<T> membership_vec(params.n_row * n_selected_clusters, handle.get_stream());

    ML::HDBSCAN::Common::PredictionData<IdxT, T> prediction_data_(
      handle, params.n_row, params.n_col, nullptr);

    transformLabels(handle, labels.data(), label_map.data(), params.n_row);

    ML::HDBSCAN::Common::generate_prediction_data(handle,
                                                  condensed_tree,
                                                  labels.data(),
                                                  inverse_label_map.data(),
                                                  n_selected_clusters,
                                                  prediction_data_);

    ML::compute_all_points_membership_vectors(handle,
                                              condensed_tree,
                                              prediction_data_,
                                              data.data(),
                                              ML::distance::DistanceType::L2SqrtExpanded,
                                              membership_vec.data());

    ASSERT_TRUE(MLCommon::devArrMatch(membership_vec.data(),
                                      params.expected_probabilities.data(),
                                      params.n_row * n_selected_clusters,
                                      MLCommon::CompareApprox<float>(1e-5),
                                      handle.get_stream().get()));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  AllPointsMembershipVectorsInputs<T, IdxT> params;
  // T score;
};

typedef AllPointsMembershipVectorsTest<float, int64_t> AllPointsMembershipVectorsTestF_Int;
TEST_P(AllPointsMembershipVectorsTestF_Int, Result) { EXPECT_TRUE(true); }

INSTANTIATE_TEST_CASE_P(AllPointsMembershipVectorsTest,
                        AllPointsMembershipVectorsTestF_Int,
                        ::testing::ValuesIn(all_points_membership_vectors_inputs));

template <typename T, typename IdxT>
class ApproximatePredictTest : public ::testing::TestWithParam<ApproximatePredictInputs<T, IdxT>> {
 public:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<ApproximatePredictInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<IdxT> condensed_parents(params.condensed_parents.size(),
                                                handle.get_stream());
    rmm::device_uvector<IdxT> condensed_children(params.condensed_children.size(),
                                                 handle.get_stream());
    rmm::device_uvector<T> condensed_lambdas(params.condensed_lambdas.size(), handle.get_stream());
    rmm::device_uvector<IdxT> condensed_sizes(params.condensed_sizes.size(), handle.get_stream());

    raft::copy(condensed_parents.data(),
               params.condensed_parents.data(),
               condensed_parents.size(),
               handle.get_stream());

    raft::copy(condensed_children.data(),
               params.condensed_children.data(),
               condensed_children.size(),
               handle.get_stream());

    raft::copy(condensed_lambdas.data(),
               params.condensed_lambdas.data(),
               condensed_lambdas.size(),
               handle.get_stream());

    raft::copy(condensed_sizes.data(),
               params.condensed_sizes.data(),
               condensed_sizes.size(),
               handle.get_stream());

    rmm::device_uvector<T> data(params.n_row * params.n_col, handle.get_stream());
    raft::copy(data.data(), params.data.data(), data.size(), handle.get_stream());

    rmm::device_uvector<T> points_to_predict(params.n_points_to_predict * params.n_col,
                                             handle.get_stream());
    raft::copy(points_to_predict.data(),
               params.points_to_predict.data(),
               points_to_predict.size(),
               handle.get_stream());

    ML::HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle,
                                                                    params.n_row,
                                                                    params.condensed_parents.size(),
                                                                    condensed_parents.data(),
                                                                    condensed_children.data(),
                                                                    condensed_lambdas.data(),
                                                                    condensed_sizes.data());

    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());

    // intermediate outputs
    rmm::device_uvector<T> stabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> inverse_label_map(0, handle.get_stream());

    int n_selected_clusters =
      ML::HDBSCAN::detail::Extract::extract_clusters(handle,
                                                     condensed_tree,
                                                     params.n_row,
                                                     labels.data(),
                                                     stabilities.data(),
                                                     probabilities.data(),
                                                     label_map.data(),
                                                     params.cluster_selection_method,
                                                     inverse_label_map,
                                                     params.allow_single_cluster,
                                                     static_cast<IdxT>(0),
                                                     params.cluster_selection_epsilon);

    rmm::device_uvector<T> core_dists{static_cast<size_t>(params.n_row), handle.get_stream().get()};
    ML::HDBSCAN::Common::PredictionData<IdxT, T> pred_data(
      handle, params.n_row, params.n_col, core_dists.data());

    auto stream = handle.get_stream().get();
    rmm::device_uvector<IdxT> mutual_reachability_indptr(params.n_row + 1, stream);
    raft::sparse::COO<T, IdxT> mutual_reachability_coo(stream,
                                                       (params.min_samples + 1) * params.n_row * 2);

    auto exec_policy     = raft::resource::get_thrust_policy(handle);
    auto new_min_samples = params.min_samples + 1;
    auto inds  = raft::make_device_matrix<IdxT, IdxT>(handle, params.n_row, new_min_samples);
    auto dists = raft::make_device_matrix<T, IdxT>(handle, params.n_row, new_min_samples);

    cuvs::neighbors::all_neighbors::all_neighbors_params all_neigh_p;
    all_neigh_p.metric                = cuvs::distance::DistanceType::L2SqrtExpanded;
    auto brute_force_p                = cuvs::neighbors::graph_build_params::brute_force_params{};
    brute_force_p.build_params.metric = cuvs::distance::DistanceType::L2SqrtExpanded;
    all_neigh_p.graph_build_params    = brute_force_p;
    cuvs::neighbors::all_neighbors::build(
      handle,
      all_neigh_p,
      raft::make_device_matrix_view<const T, int64_t>(data.data(), params.n_row, params.n_col),
      inds.view(),
      dists.view(),
      raft::make_device_vector_view<T>(core_dists.data(), params.n_row),
      1.0);

    // self-loops get max distance
    rmm::device_uvector<IdxT> coo_rows(new_min_samples * params.n_row, stream);
    auto coo_rows_counting_itr = thrust::make_counting_iterator<IdxT>(0);
    thrust::transform(exec_policy,
                      coo_rows_counting_itr,
                      coo_rows_counting_itr + (params.n_row * new_min_samples),
                      coo_rows.data(),
                      [new_min_samples] __device__(IdxT c) -> IdxT { return c / new_min_samples; });

    raft::sparse::linalg::symmetrize(handle,
                                     coo_rows.data(),
                                     inds.data_handle(),
                                     dists.data_handle(),
                                     static_cast<IdxT>(params.n_row),
                                     static_cast<IdxT>(params.n_row),
                                     static_cast<size_t>(new_min_samples * params.n_row),
                                     mutual_reachability_coo);

    raft::sparse::convert::sorted_coo_to_csr(mutual_reachability_coo.rows(),
                                             mutual_reachability_coo.nnz,
                                             mutual_reachability_indptr.data(),
                                             (int)params.n_row + 1,
                                             stream);

    auto transform_in =
      thrust::make_zip_iterator(cuda::std::make_tuple(mutual_reachability_coo.rows(),
                                                      mutual_reachability_coo.cols(),
                                                      mutual_reachability_coo.vals()));

    thrust::transform(exec_policy,
                      transform_in,
                      transform_in + mutual_reachability_coo.nnz,
                      mutual_reachability_coo.vals(),
                      [=] __device__(const cuda::std::tuple<IdxT, IdxT, T>& tup) {
                        return cuda::std::get<0>(tup) == cuda::std::get<1>(tup)
                                 ? std::numeric_limits<T>::max()
                                 : cuda::std::get<2>(tup);
                      });

    transformLabels(handle, labels.data(), label_map.data(), params.n_row);
    ML::HDBSCAN::Common::generate_prediction_data(handle,
                                                  condensed_tree,
                                                  labels.data(),
                                                  inverse_label_map.data(),
                                                  n_selected_clusters,
                                                  pred_data);

    // outputs
    rmm::device_uvector<IdxT> out_labels(params.n_points_to_predict, handle.get_stream());
    rmm::device_uvector<T> out_probabilities(params.n_points_to_predict, handle.get_stream());

    ML::out_of_sample_predict(handle,
                              condensed_tree,
                              pred_data,
                              const_cast<float*>(data.data()),
                              labels.data(),
                              const_cast<float*>(points_to_predict.data()),
                              (size_t)params.n_points_to_predict,
                              ML::distance::DistanceType::L2SqrtExpanded,
                              params.min_samples,
                              out_labels.data(),
                              out_probabilities.data());

    handle.sync_stream(handle.get_stream().get());
    cudaDeviceSynchronize();

    ASSERT_TRUE(MLCommon::devArrMatch(out_labels.data(),
                                      params.expected_labels.data(),
                                      params.n_points_to_predict,
                                      MLCommon::Compare<int>(),
                                      handle.get_stream().get()));

    ASSERT_TRUE(MLCommon::devArrMatch(out_probabilities.data(),
                                      params.expected_probabilities.data(),
                                      params.n_points_to_predict,
                                      MLCommon::CompareApprox<float>(1e-2),
                                      handle.get_stream().get()));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  ApproximatePredictInputs<T, IdxT> params;
  // T score;
};

typedef ApproximatePredictTest<float, int64_t> ApproximatePredictTestF_Int;
TEST_P(ApproximatePredictTestF_Int, Result) { EXPECT_TRUE(true); }

INSTANTIATE_TEST_CASE_P(ApproximatePredictTest,
                        ApproximatePredictTestF_Int,
                        ::testing::ValuesIn(approximate_predict_inputs));

template <typename T, typename IdxT>
class MembershipVectorTest : public ::testing::TestWithParam<MembershipVectorInputs<T, IdxT>> {
 public:
  void basicTest()
  {
    raft::handle_t handle;

    params = ::testing::TestWithParam<MembershipVectorInputs<T, IdxT>>::GetParam();

    rmm::device_uvector<IdxT> condensed_parents(params.condensed_parents.size(),
                                                handle.get_stream());
    rmm::device_uvector<IdxT> condensed_children(params.condensed_children.size(),
                                                 handle.get_stream());
    rmm::device_uvector<T> condensed_lambdas(params.condensed_lambdas.size(), handle.get_stream());
    rmm::device_uvector<IdxT> condensed_sizes(params.condensed_sizes.size(), handle.get_stream());

    raft::copy(condensed_parents.data(),
               params.condensed_parents.data(),
               condensed_parents.size(),
               handle.get_stream());

    raft::copy(condensed_children.data(),
               params.condensed_children.data(),
               condensed_children.size(),
               handle.get_stream());

    raft::copy(condensed_lambdas.data(),
               params.condensed_lambdas.data(),
               condensed_lambdas.size(),
               handle.get_stream());

    raft::copy(condensed_sizes.data(),
               params.condensed_sizes.data(),
               condensed_sizes.size(),
               handle.get_stream());

    rmm::device_uvector<T> data(params.n_row * params.n_col, handle.get_stream());
    raft::copy(data.data(), params.data.data(), data.size(), handle.get_stream());

    rmm::device_uvector<T> points_to_predict(params.n_points_to_predict * params.n_col,
                                             handle.get_stream());
    raft::copy(points_to_predict.data(),
               params.points_to_predict.data(),
               points_to_predict.size(),
               handle.get_stream());

    ML::HDBSCAN::Common::CondensedHierarchy<IdxT, T> condensed_tree(handle,
                                                                    params.n_row,
                                                                    params.condensed_parents.size(),
                                                                    condensed_parents.data(),
                                                                    condensed_children.data(),
                                                                    condensed_lambdas.data(),
                                                                    condensed_sizes.data());

    rmm::device_uvector<IdxT> label_map(params.n_row, handle.get_stream());

    // intermediate outputs
    rmm::device_uvector<T> stabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<T> probabilities(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> labels(params.n_row, handle.get_stream());
    rmm::device_uvector<IdxT> inverse_label_map(0, handle.get_stream());

    int n_selected_clusters =
      ML::HDBSCAN::detail::Extract::extract_clusters(handle,
                                                     condensed_tree,
                                                     params.n_row,
                                                     labels.data(),
                                                     stabilities.data(),
                                                     probabilities.data(),
                                                     label_map.data(),
                                                     params.cluster_selection_method,
                                                     inverse_label_map,
                                                     params.allow_single_cluster,
                                                     static_cast<IdxT>(0),
                                                     params.cluster_selection_epsilon);

    rmm::device_uvector<T> membership_vec(params.n_points_to_predict * n_selected_clusters,
                                          handle.get_stream().get());

    rmm::device_uvector<T> core_dists{static_cast<size_t>(params.n_row), handle.get_stream().get()};
    ML::HDBSCAN::Common::PredictionData<IdxT, T> prediction_data_(
      handle, params.n_row, params.n_col, core_dists.data());

    auto stream = handle.get_stream().get();
    rmm::device_uvector<IdxT> mutual_reachability_indptr(params.n_row + 1, stream);
    raft::sparse::COO<T, IdxT> mutual_reachability_coo(stream,
                                                       (params.min_samples + 1) * params.n_row * 2);

    auto exec_policy     = raft::resource::get_thrust_policy(handle);
    auto new_min_samples = params.min_samples + 1;
    auto inds  = raft::make_device_matrix<IdxT, IdxT>(handle, params.n_row, new_min_samples);
    auto dists = raft::make_device_matrix<T, IdxT>(handle, params.n_row, new_min_samples);

    cuvs::neighbors::all_neighbors::all_neighbors_params all_neigh_p;
    all_neigh_p.metric                = cuvs::distance::DistanceType::L2SqrtExpanded;
    auto brute_force_p                = cuvs::neighbors::graph_build_params::brute_force_params{};
    brute_force_p.build_params.metric = cuvs::distance::DistanceType::L2SqrtExpanded;
    all_neigh_p.graph_build_params    = brute_force_p;
    cuvs::neighbors::all_neighbors::build(
      handle,
      all_neigh_p,
      raft::make_device_matrix_view<const T, int64_t>(data.data(), params.n_row, params.n_col),
      inds.view(),
      dists.view(),
      raft::make_device_vector_view<T>(core_dists.data(), params.n_row),
      1.0);

    // self-loops get max distance
    rmm::device_uvector<IdxT> coo_rows(new_min_samples * params.n_row, stream);
    auto coo_rows_counting_itr = thrust::make_counting_iterator<IdxT>(0);
    thrust::transform(exec_policy,
                      coo_rows_counting_itr,
                      coo_rows_counting_itr + (params.n_row * new_min_samples),
                      coo_rows.data(),
                      [new_min_samples] __device__(IdxT c) -> IdxT { return c / new_min_samples; });

    raft::sparse::linalg::symmetrize(handle,
                                     coo_rows.data(),
                                     inds.data_handle(),
                                     dists.data_handle(),
                                     static_cast<IdxT>(params.n_row),
                                     static_cast<IdxT>(params.n_row),
                                     static_cast<size_t>(new_min_samples * params.n_row),
                                     mutual_reachability_coo);

    raft::sparse::convert::sorted_coo_to_csr(mutual_reachability_coo.rows(),
                                             mutual_reachability_coo.nnz,
                                             mutual_reachability_indptr.data(),
                                             (int)params.n_row + 1,
                                             stream);

    auto transform_in =
      thrust::make_zip_iterator(cuda::std::make_tuple(mutual_reachability_coo.rows(),
                                                      mutual_reachability_coo.cols(),
                                                      mutual_reachability_coo.vals()));

    thrust::transform(exec_policy,
                      transform_in,
                      transform_in + mutual_reachability_coo.nnz,
                      mutual_reachability_coo.vals(),
                      [=] __device__(const cuda::std::tuple<IdxT, IdxT, T>& tup) {
                        return cuda::std::get<0>(tup) == cuda::std::get<1>(tup)
                                 ? std::numeric_limits<T>::max()
                                 : cuda::std::get<2>(tup);
                      });

    transformLabels(handle, labels.data(), label_map.data(), params.n_row);

    ML::HDBSCAN::Common::generate_prediction_data(handle,
                                                  condensed_tree,
                                                  labels.data(),
                                                  inverse_label_map.data(),
                                                  n_selected_clusters,
                                                  prediction_data_);

    ML::compute_membership_vector(handle,
                                  condensed_tree,
                                  prediction_data_,
                                  data.data(),
                                  points_to_predict.data(),
                                  params.n_points_to_predict,
                                  params.min_samples,
                                  ML::distance::DistanceType::L2SqrtExpanded,
                                  membership_vec.data());

    ASSERT_TRUE(MLCommon::devArrMatch(membership_vec.data(),
                                      params.expected_probabilities.data(),
                                      params.n_points_to_predict * n_selected_clusters,
                                      MLCommon::CompareApprox<float>(1e-4),
                                      handle.get_stream().get()));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  MembershipVectorInputs<T, IdxT> params;
  // T score;
};

typedef MembershipVectorTest<float, int64_t> MembershipVectorTestF_Int;
TEST_P(MembershipVectorTestF_Int, Result) { EXPECT_TRUE(true); }

INSTANTIATE_TEST_CASE_P(MembershipVectorTest,
                        MembershipVectorTestF_Int,
                        ::testing::ValuesIn(membership_vector_inputs));

}  // namespace HDBSCAN
}  // end namespace ML
