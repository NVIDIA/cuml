/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

namespace MLCommon {

/**
 * @defgroup SmemStores Shared memory store operations
 * @{
 * @brief Stores to shared memory (both vectorized and non-vectorized forms)
 * @param[out] addr shared memory address
 * @param[in]  x    data to be stored at this address
 */
DI void sts(float* addr, const float& x) { *addr = x; }
DI void sts(float* addr, const float (&x)[1]) { *addr = x[0]; }
DI void sts(float* addr, const float (&x)[2])
{
  float2 v2 = make_float2(x[0], x[1]);
  auto* s2  = reinterpret_cast<float2*>(addr);
  *s2       = v2;
}
DI void sts(float* addr, const float (&x)[4])
{
  float4 v4 = make_float4(x[0], x[1], x[2], x[3]);
  auto* s4  = reinterpret_cast<float4*>(addr);
  *s4       = v4;
}
DI void sts(double* addr, const double& x) { *addr = x; }
DI void sts(double* addr, const double (&x)[1]) { *addr = x[0]; }
DI void sts(double* addr, const double (&x)[2])
{
  double2 v2 = make_double2(x[0], x[1]);
  auto* s2   = reinterpret_cast<double2*>(addr);
  *s2        = v2;
}
/** @} */

/**
 * @defgroup SmemLoads Shared memory load operations
 * @{
 * @brief Loads from shared memory (both vectorized and non-vectorized forms)
 * @param[out] x    the data to be loaded
 * @param[in]  addr shared memory address from where to load
 */
DI void lds(float& x, float* addr) { x = *addr; }
DI void lds(float (&x)[1], float* addr) { x[0] = *addr; }
DI void lds(float (&x)[2], float* addr)
{
  auto* s2 = reinterpret_cast<float2*>(addr);
  auto v2  = *s2;
  x[0]     = v2.x;
  x[1]     = v2.y;
}
DI void lds(float (&x)[4], float* addr)
{
  auto* s4 = reinterpret_cast<float4*>(addr);
  auto v4  = *s4;
  x[0]     = v4.x;
  x[1]     = v4.y;
  x[2]     = v4.z;
  x[3]     = v4.w;
}
DI void lds(double& x, double* addr) { x = *addr; }
DI void lds(double (&x)[1], double* addr) { x[0] = *addr; }
DI void lds(double (&x)[2], double* addr)
{
  auto* s2 = reinterpret_cast<double2*>(addr);
  auto v2  = *s2;
  x[0]     = v2.x;
  x[1]     = v2.y;
}
/** @} */

/**
 * @defgroup GlobalLoads Global cached load operations
 * @{
 * @brief Load from global memory with caching at L2 level (bypassing L1)
 * @param[out] x    data to be loaded from global memory
 * @param[in]  addr address in global memory from where to load
 */
DI void ldg(float& x, const float* addr)
{
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(addr));
}
DI void ldg(float (&x)[1], const float* addr)
{
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x[0]) : "l"(addr));
}
DI void ldg(float (&x)[2], const float* addr)
{
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(addr));
}
DI void ldg(float (&x)[4], const float* addr)
{
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(addr));
}
DI void ldg(double& x, const double* addr)
{
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x) : "l"(addr));
}
DI void ldg(double (&x)[1], const double* addr)
{
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x[0]) : "l"(addr));
}
DI void ldg(double (&x)[2], const double* addr)
{
  asm volatile("ld.global.cg.v2.f64 {%0, %1}, [%2];" : "=d"(x[0]), "=d"(x[1]) : "l"(addr));
}
/** @} */

}  // namespace MLCommon
