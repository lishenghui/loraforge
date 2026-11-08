// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Shenghui Li
// Adapted from probe/tools/llm_fleet/sglang_plugin/probe_sglang_gemv/fast_decode.cu
// at 5fd6ae060d106066b306959f9a34b726a281ef4c. See docs/provenance.md.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
constexpr int WARP_WIDTH = 64;
#define ROW_SHFL_DOWN(value, delta) __shfl_down(value, delta, 64)
#else
#include <cuda_runtime.h>
constexpr int WARP_WIDTH = 32;
#define ROW_SHFL_DOWN(value, delta) __shfl_down_sync(0xffffffffu, value, delta)
#endif

constexpr int SHRINK_THREADS = 256;
constexpr int SHRINK_WARPS = SHRINK_THREADS / WARP_WIDTH;

__device__ __forceinline__ float bf16_to_float(uint16_t x) {
  return __uint_as_float(uint32_t(x) << 16);
}

__device__ __forceinline__ uint16_t round_bf16(float x) {
  uint32_t u = __float_as_uint(x);
  if ((u & 0x7f800000u) != 0x7f800000u) {
    u += 0x7fffu + ((u >> 16) & 1u);
  }
  return uint16_t(u >> 16);
}

__global__ void row_shrink_kernel(
    const uint16_t* x, const uint16_t* a, uint16_t* z,
    const int* map, const int* ranks, int K, int S, int R, int max_rank,
    int64_t x_stride) {
  const int row = blockIdx.x;
  const int slot = map[row];
  if (slot < 0) return;
  const int rank = ranks[slot];
  if (!rank) return;
  const int lane = threadIdx.x % WARP_WIDTH;
  const int warp = threadIdx.x / WARP_WIDTH;
  const int j = blockIdx.y * SHRINK_WARPS + warp;
  const int slice = j / max_rank;
  const int k = j % max_rank;
  if (slice >= S || k >= rank) return;
  const int column = slice * rank + k;
  const uint16_t* ap = a + (int64_t(slot) * S * R + column) * K;
  const uint16_t* xp = x + int64_t(row) * x_stride;
  float sum = 0;
  for (int t = lane * 8; t < K; t += WARP_WIDTH * 8) {
    const uint4 av = *reinterpret_cast<const uint4*>(ap + t);
    const uint4 xv = *reinterpret_cast<const uint4*>(xp + t);
    const auto* aa = reinterpret_cast<const uint16_t*>(&av);
    const auto* xx = reinterpret_cast<const uint16_t*>(&xv);
#pragma unroll
    for (int i = 0; i < 8; ++i) sum += bf16_to_float(aa[i]) * bf16_to_float(xx[i]);
  }
#pragma unroll
  for (int d = WARP_WIDTH / 2; d > 0; d >>= 1) sum += ROW_SHFL_DOWN(sum, d);
  if (lane == 0) z[int64_t(row) * S * R + column] = round_bf16(sum);
}

template<int BR>
__device__ __forceinline__ void expand_tile(
    const uint16_t* z, const uint16_t* bp, uint16_t* y,
    int N, int S, int R, int64_t y_stride, int n, int T, int rank,
    const int* rows, float* zs, const int* slice_offsets, float scale) {
  for (int i = threadIdx.x; i < T * S * BR; i += blockDim.x) {
    const int t = i / (S * BR);
    const int rem = i % (S * BR);
    const int slice = rem / BR;
    const int k = rem % BR;
    zs[(t * 3 + slice) * 64 + k] = k < rank
        ? bf16_to_float(z[int64_t(rows[t]) * S * R + slice * rank + k]) : 0.f;
  }
  __syncthreads();
  if (n < N) {
    int slice = 0;
    for (int i = 1; i < S; ++i) if (n >= slice_offsets[i]) slice = i;
    float b[BR];
    bp += int64_t(n) * BR;
    if constexpr (BR >= 8) {
#pragma unroll
      for (int q = 0; q < BR; q += 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(bp + q);
        const auto* values = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) b[q + j] = bf16_to_float(values[j]);
      }
    } else if constexpr (BR == 4) {
      const uint2 v = *reinterpret_cast<const uint2*>(bp);
      const auto* values = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = bf16_to_float(values[j]);
    } else if constexpr (BR == 2) {
      const uint32_t v = *reinterpret_cast<const uint32_t*>(bp);
      const auto* values = reinterpret_cast<const uint16_t*>(&v);
      b[0] = bf16_to_float(values[0]);
      b[1] = bf16_to_float(values[1]);
    } else {
      b[0] = bf16_to_float(bp[0]);
    }
    for (int t = 0; t < T; ++t) {
      float acc = 0;
#pragma unroll
      for (int k = 0; k < BR; ++k) acc += zs[(t * 3 + slice) * 64 + k] * b[k];
      uint16_t* yp = y + int64_t(rows[t]) * y_stride + n;
      // Preserve the source's BF16 scaled-delta rounding before base addition.
      *yp = round_bf16(bf16_to_float(*yp) + bf16_to_float(round_bf16(acc * scale)));
    }
  }
  __syncthreads();
}

__global__ void row_expand_kernel(
    const uint16_t* z, const uint16_t* B, const int64_t* b_offsets,
    const int* buckets, uint16_t* y, const int* indptr, const int* slots,
    const int* permutation, const int* ranks, const float* scales,
    const int* slice_offsets, const int* segment_count,
    int N, int S, int R, int64_t y_stride) {
  __shared__ int rows[16];
  __shared__ float zs[16 * 3 * 64];
  const int column_blocks = (N + blockDim.x - 1) / blockDim.x;
  const int items = (*segment_count) * column_blocks;
  for (int item = blockIdx.x; item < items; item += gridDim.x) {
    const int segment = item / column_blocks;
    const int column_block = item % column_blocks;
    const int start = indptr[segment];
    const int end = indptr[segment + 1];
    const int T = end - start;
    const int slot = slots[segment];
    if (slot < 0) continue;
    const int rank = ranks[slot];
    if (T <= 0 || rank <= 0) continue;
    if (T > 16 || rank > 64) return;  // Caller metadata contract.
    if (threadIdx.x < 16) {
      rows[threadIdx.x] = threadIdx.x < T ? permutation[start + threadIdx.x] : 0;
    }
    __syncthreads();
    const int n = column_block * blockDim.x + threadIdx.x;
    const int bucket = buckets[slot];
    const uint16_t* bp = B + b_offsets[slot];
    const float scale = scales[slot];
#define EXPAND_CASE(BR) \
    case BR: expand_tile<BR>(z, bp, y, N, S, R, y_stride, n, T, rank, rows, zs, \
                             slice_offsets, scale); break
    switch (bucket) {
      EXPAND_CASE(1);
      EXPAND_CASE(2);
      EXPAND_CASE(4);
      EXPAND_CASE(8);
      EXPAND_CASE(16);
      EXPAND_CASE(32);
      EXPAND_CASE(64);
    }
#undef EXPAND_CASE
  }
}

// Python validates shapes, strides, alignment and aliasing without reading device
// values. Private bindings additionally reject cross-device and dtype mistakes.
static void same_device(const torch::Tensor& reference, const torch::Tensor& t,
                        at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.device() == reference.device(), "device mismatch");
  TORCH_CHECK(t.scalar_type() == dtype, "dtype mismatch");
}

void shrink(torch::Tensor x, torch::Tensor a, torch::Tensor z,
            torch::Tensor map, torch::Tensor ranks, int64_t slices, int64_t max_rank) {
  for (const auto& t : {x, a, z}) same_device(x, t, at::kBFloat16);
  for (const auto& t : {map, ranks}) same_device(x, t, at::kInt);
  if (x.size(0) == 0) return;
  c10::cuda::CUDAGuard guard(x.device());
  row_shrink_kernel<<<
      dim3(x.size(0), (slices * max_rank + SHRINK_WARPS - 1) / SHRINK_WARPS),
      dim3(SHRINK_THREADS), 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      reinterpret_cast<const uint16_t*>(a.data_ptr()),
      reinterpret_cast<uint16_t*>(z.data_ptr()), map.data_ptr<int>(), ranks.data_ptr<int>(),
      x.size(1), slices, a.size(1) / slices, max_rank, x.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void expand(torch::Tensor z, torch::Tensor B, torch::Tensor b_offsets,
            torch::Tensor buckets, torch::Tensor y, torch::Tensor indptr,
            torch::Tensor slots, torch::Tensor permutation, torch::Tensor ranks,
            torch::Tensor scales, torch::Tensor slice_offsets, torch::Tensor segment_count,
            int64_t N, int64_t R, int64_t blocks, int64_t threads) {
  for (const auto& t : {z, B, y}) same_device(z, t, at::kBFloat16);
  for (const auto& t : {buckets, indptr, slots, permutation, ranks, slice_offsets, segment_count}) {
    same_device(z, t, at::kInt);
  }
  same_device(z, b_offsets, at::kLong);
  same_device(z, scales, at::kFloat);
  if (z.size(0) == 0) return;
  c10::cuda::CUDAGuard guard(z.device());
  row_expand_kernel<<<dim3(blocks), dim3(threads), 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint16_t*>(z.data_ptr()),
      reinterpret_cast<const uint16_t*>(B.data_ptr()), b_offsets.data_ptr<int64_t>(),
      buckets.data_ptr<int>(), reinterpret_cast<uint16_t*>(y.data_ptr()),
      indptr.data_ptr<int>(), slots.data_ptr<int>(), permutation.data_ptr<int>(),
      ranks.data_ptr<int>(), scales.data_ptr<float>(), slice_offsets.data_ptr<int>(),
      segment_count.data_ptr<int>(), N, slice_offsets.numel() - 1, R, y.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shrink", &shrink);
  m.def("expand", &expand);
}
