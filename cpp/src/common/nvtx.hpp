/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/nvtx.hpp>

namespace ML {

/**
 * @brief Push a named nvtx range
 * @param name range name
 * @param stream unused
 */
[[deprecated("Use new raft::common::nvtx::push_range from <raft/core/nvtx.hpp>")]] inline void
PUSH_RANGE(const char* name, cudaStream_t stream)
{
  raft::common::nvtx::push_range(name);
}

/**
 * @brief Pop the latest nvtx range
 * @param stream unused
 */
[[deprecated("Use new raft::common::nvtx::pop_range from <raft/core/nvtx.hpp>")]] inline void
POP_RANGE(cudaStream_t stream)
{
  raft::common::nvtx::pop_range();
}

/**
 * @brief Push a named nvtx range
 * @param name range name
 */
[[deprecated("Use new raft::common::nvtx::push_range from <raft/core/nvtx.hpp>")]] inline void
PUSH_RANGE(const char* name)
{
  raft::common::nvtx::push_range(name);
}

/** Pop the latest range */
[[deprecated("Use new raft::common::nvtx::pop_range from <raft/core/nvtx.hpp>")]] inline void
POP_RANGE()
{
  raft::common::nvtx::pop_range();
}

}  // end namespace ML
