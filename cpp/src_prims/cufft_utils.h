/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/error.hpp>

#include <cufft.h>

// TODO move to raft https://github.com/NVIDIA/raft/issues/91
namespace raft {

/**
 * @brief Exception thrown when a cuFFT error is encountered.
 */
struct cufft_error : public raft::exception {
  explicit cufft_error(char const* const message) : raft::exception(message) {}
  explicit cufft_error(std::string const& message) : raft::exception(message) {}
};

const char* getCufftErrStr(cufftResult status)
{
  // https://docs.nvidia.com/cuda/cufft/index.html#cufftresult
  switch (status) {
    case CUFFT_SUCCESS: return "The cuFFT operation was successful.";
    case CUFFT_INVALID_PLAN: return "cuFFT was passed an invalid plan handle.";
    case CUFFT_ALLOC_FAILED: return "cuFFT failed to allocate GPU or CPU memory.";
    case CUFFT_INVALID_VALUE: return "User specified an invalid pointer or parameter.";
    case CUFFT_INTERNAL_ERROR: return "Driver or internal cuFFT library error.";
    case CUFFT_EXEC_FAILED: return "Failed to execute an FFT on the GPU.";
    case CUFFT_SETUP_FAILED: return "The cuFFT library failed to initialize.";
    case CUFFT_INVALID_SIZE: return "User specified an invalid transform size.";
#if defined(CUDART_VERSION) && CUDART_VERSION < 13000
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "Missing parameters in call.";
#endif
    case CUFFT_INVALID_DEVICE:
      return "Execution of a plan was on different GPU than plan creation.";
#if defined(CUDART_VERSION) && CUDART_VERSION < 13000
    case CUFFT_PARSE_ERROR: return "Internal plan database error.";
#endif
    case CUFFT_NO_WORKSPACE: return "No workspace has been provided prior to plan execution.";
    case CUFFT_NOT_IMPLEMENTED:
      return "Function does not implement functionality for parameters given.";
    case CUFFT_NOT_SUPPORTED: return "Operation is not supported for parameters given.";
    default: return "Unknown error.";
  }
}

/**
 * @brief Error checking macro for cuFFT functions.
 *
 * Invokes a cuFFT function. If the call does not return CUFFT_SUCCESS, throws
 * an exception detailing the error that occurred.
 */
#define CUFFT_TRY(call)                             \
  do {                                              \
    const cufftResult status = call;                \
    if (status != CUFFT_SUCCESS) {                  \
      std::string msg{};                            \
      SET_ERROR_MSG(msg,                            \
                    "cuFFT error encountered at: ", \
                    "call='%s', Reason=%s",         \
                    #call,                          \
                    raft::getCufftErrStr(status));  \
      throw raft::cufft_error(msg);                 \
    }                                               \
  } while (0)

class CuFFTHandle {
 public:
  CuFFTHandle(cudaStream_t stream)
  {
    CUFFT_TRY(cufftCreate(&handle));
    CUFFT_TRY(cufftSetStream(handle, stream));
  }
  ~CuFFTHandle() { cufftDestroy(handle); }
  operator cufftHandle() const { return handle; }

 private:
  cufftHandle handle;
};

}  // namespace raft
