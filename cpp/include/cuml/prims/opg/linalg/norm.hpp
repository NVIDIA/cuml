/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../matrix/data.hpp"
#include "../matrix/part_descriptor.hpp"

#include <cuml/common/export.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

namespace CUML_EXPORT MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief performs MNMG column-wise L2 norm calculation.
 */
void colNorm2(const raft::handle_t& handle,
              Matrix::Data<double>& out,
              const std::vector<Matrix::Data<double>*>& in,
              const Matrix::PartDescriptor& inDesc,
              cudaStream_t* streams,
              int n_streams);

void colNorm2(const raft::handle_t& handle,
              Matrix::Data<float>& out,
              const std::vector<Matrix::Data<float>*>& in,
              const Matrix::PartDescriptor& inDesc,
              cudaStream_t* streams,
              int n_streams);

void colNorm2NoSeq(const raft::handle_t& handle,
                   Matrix::Data<double>& out,
                   const std::vector<Matrix::Data<double>*>& in,
                   const Matrix::PartDescriptor& inDesc,
                   cudaStream_t* streams,
                   int n_streams);

void colNorm2NoSeq(const raft::handle_t& handle,
                   Matrix::Data<float>& out,
                   const std::vector<Matrix::Data<float>*>& in,
                   const Matrix::PartDescriptor& inDesc,
                   cudaStream_t* streams,
                   int n_streams);

}  // end namespace opg
}  // end namespace LinAlg
}  // namespace CUML_EXPORT MLCommon
