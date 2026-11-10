/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>
#include <cuml/linear_model/glm.hpp>
#include <cuml/prims/opg/matrix/data.hpp>
#include <cuml/prims/opg/matrix/part_descriptor.hpp>

namespace CUML_EXPORT ML {
namespace CD {
namespace opg {

/**
 * @brief performs MNMG coordinate descent fit (elastic net / lasso)
 * @param[in] handle: the internal cuml handle object
 * @param[in] input_data: vector holding all partitions for that rank
 * @param[in] input_desc: PartDescriptor object for the input
 * @param[in] labels: labels data
 * @param[out] coef: learned regression coefficients
 * @param[out] intercept: intercept value
 * @param[in] fit_intercept: fit intercept or not
 * @param[in] epochs: number of epochs
 * @param[in] alpha: regularization strength
 * @param[in] l1_ratio: l1 ratio
 * @param[in] shuffle: whether to shuffle the data
 * @param[in] tol: tolerance for early stopping during fitting
 * @param[in] verbose
 * @return n_iter: the number of solver iterations run
 */
int fit(raft::handle_t& handle,
        std::vector<MLCommon::Matrix::Data<float>*>& input_data,
        MLCommon::Matrix::PartDescriptor& input_desc,
        std::vector<MLCommon::Matrix::Data<float>*>& labels,
        float* coef,
        float* intercept,
        bool fit_intercept,
        int epochs,
        float alpha,
        float l1_ratio,
        bool shuffle,
        float tol,
        bool verbose);

int fit(raft::handle_t& handle,
        std::vector<MLCommon::Matrix::Data<double>*>& input_data,
        MLCommon::Matrix::PartDescriptor& input_desc,
        std::vector<MLCommon::Matrix::Data<double>*>& labels,
        double* coef,
        double* intercept,
        bool fit_intercept,
        int epochs,
        double alpha,
        double l1_ratio,
        bool shuffle,
        double tol,
        bool verbose);

/**
 * @brief performs MNMG prediction for coordinate descent models
 * @param[in] handle: the internal cuml handle object
 * @param[in] rank_sizes: includes all the partition size information for the rank
 * @param[in] n_parts: number of partitions
 * @param[in] input: input data
 * @param[in] n_rows: number of rows of input data
 * @param[in] n_cols: number of cols of input data
 * @param[in] coef: learned regression coefficients
 * @param[in] intercept: the fit intercept
 * @param[out] preds: predictions
 * @param[in] verbose
 */
void predict(raft::handle_t& handle,
             MLCommon::Matrix::RankSizePair** rank_sizes,
             size_t n_parts,
             MLCommon::Matrix::Data<float>** input,
             size_t n_rows,
             size_t n_cols,
             float* coef,
             float intercept,
             MLCommon::Matrix::Data<float>** preds,
             bool verbose);

void predict(raft::handle_t& handle,
             MLCommon::Matrix::RankSizePair** rank_sizes,
             size_t n_parts,
             MLCommon::Matrix::Data<double>** input,
             size_t n_rows,
             size_t n_cols,
             double* coef,
             double intercept,
             MLCommon::Matrix::Data<double>** preds,
             bool verbose);

};  // end namespace opg
};  // namespace CD
};  // end namespace CUML_EXPORT ML
