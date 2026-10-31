// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Shenghui Li
// Derived from lishenghui/probe at 5fd6ae060d106066b306959f9a34b726a281ef4c.
// Modified for owned state, device guards, and current-stream execution.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

// Batched symmetric eigendecomposition (FP64) by parallel cyclic Jacobi. Used on AMD/ROCm, where
// torch.linalg.eigh on a batch of small matrices is slow; CUDA keeps torch.linalg.eigh.
//
// One workgroup per matrix. A sweep has n-1 rounds; each round rotates n/2 disjoint (p, q) pairs chosen by
// round-robin tournament ordering, so the rotations of one round commute and run in parallel:
//   1. one thread per pair computes (c, s) that zero a_pq           (Numerical Recipes convention, A' = J^T A J)
//   2. row update    A[p,:] = c A[p,:] - s A[q,:],  A[q,:] = s A[p,:] + c A[q,:]
//   3. column update A[:,p] = c A[:,p] - s A[:,q],  A[:,q] = s A[:,p] + c A[:,q], same on V
// Sweeps stop when the off-diagonal Frobenius norm <= tol * ||A||_F (or after max_sweeps).
// n is padded to even with a zero row/column; that index never rotates (a_pq = 0) and is dropped by the caller.
// A and V live in LDS when both fit (2 n^2 doubles), otherwise in a global workspace (one slice per matrix).

__device__ __forceinline__ void pair_of(int r, int k, int n, int& p, int& q) {
    int m = n - 1;
    p = (k == 0) ? 0 : 1 + (r + k - 1) % m;
    q = 1 + (r + m - 1 - k) % m;
    if (p > q) { int t = p; p = q; q = t; }
}

__device__ double block_sum(double v, double* red) {
    int t = threadIdx.x;
    red[t] = v; __syncthreads();
    for (int d = blockDim.x / 2; d > 0; d >>= 1) { if (t < d) red[t] += red[t + d]; __syncthreads(); }
    double s = red[0]; __syncthreads();
    return s;
}

__global__ void jacobi_kernel(const double* __restrict__ Ain, double* __restrict__ Wout, double* __restrict__ Vout,
                              double* __restrict__ work, int n, int lds, int max_sweeps, double tol, int* __restrict__ sweeps_out) {
    extern __shared__ double sm[];
    const int nn = n * n, half = n / 2, t = threadIdx.x;
    double *A, *V, *cs, *red;
    if (lds) { A = sm; V = sm + nn; cs = V + nn; }
    else { A = work + (size_t)blockIdx.x * 2 * nn; V = A + nn; cs = sm; }
    red = cs + 2 * half;
    const double* src = Ain + (size_t)blockIdx.x * nn;
    for (int i = t; i < nn; i += blockDim.x) { A[i] = src[i]; V[i] = (i / n == i % n) ? 1.0 : 0.0; }
    __syncthreads();

    double tot = 0;
    for (int i = t; i < nn; i += blockDim.x) tot += A[i] * A[i];
    tot = block_sum(tot, red);
    int sweep = 0;
    for (; sweep < max_sweeps; ++sweep) {
        double off = 0;
        for (int i = t; i < nn; i += blockDim.x) if (i / n != i % n) off += A[i] * A[i];
        off = block_sum(off, red);
        if (off <= tol * tol * tot) break;
        for (int r = 0; r < n - 1; ++r) {
            for (int k = t; k < half; k += blockDim.x) {
                int p, q; pair_of(r, k, n, p, q);
                double apq = A[p * n + q], c = 1.0, s = 0.0;
                if (fabs(apq) > 1e-300) {
                    double theta = (A[q * n + q] - A[p * n + p]) / (2.0 * apq);
                    double tt = (theta >= 0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                    c = 1.0 / sqrt(tt * tt + 1.0); s = tt * c;
                }
                cs[2 * k] = c; cs[2 * k + 1] = s;
            }
            __syncthreads();
            for (int idx = t; idx < half * n; idx += blockDim.x) {     // rows
                int k = idx / n, j = idx % n, p, q; pair_of(r, k, n, p, q);
                double c = cs[2 * k], s = cs[2 * k + 1], ap = A[p * n + j], aq = A[q * n + j];
                A[p * n + j] = c * ap - s * aq; A[q * n + j] = s * ap + c * aq;
            }
            __syncthreads();
            for (int idx = t; idx < half * n; idx += blockDim.x) {     // columns of A and V
                int k = idx / n, i = idx % n, p, q; pair_of(r, k, n, p, q);
                double c = cs[2 * k], s = cs[2 * k + 1];
                double ap = A[i * n + p], aq = A[i * n + q];
                A[i * n + p] = c * ap - s * aq; A[i * n + q] = s * ap + c * aq;
                double vp = V[i * n + p], vq = V[i * n + q];
                V[i * n + p] = c * vp - s * vq; V[i * n + q] = s * vp + c * vq;
            }
            __syncthreads();
        }
    }
    double* W = Wout + (size_t)blockIdx.x * n; double* Vo = Vout + (size_t)blockIdx.x * nn;
    for (int i = t; i < n; i += blockDim.x) W[i] = A[i * n + i];
    for (int i = t; i < nn; i += blockDim.x) Vo[i] = V[i];
    if (t == 0) sweeps_out[blockIdx.x] = sweep;
}

// G [batch, n, n] FP64 symmetric, n even. Returns {W [batch, n] (unsorted), V [batch, n, n] (columns), sweeps [batch]}.
std::vector<torch::Tensor> jacobi_eigh(torch::Tensor G, int64_t max_sweeps, double tol) {
    TORCH_CHECK(G.is_cuda() && G.scalar_type() == torch::kFloat64 && G.dim() == 3 && G.size(1) == G.size(2) && G.size(1) % 2 == 0,
                "expected CUDA/HIP fp64 [batch, n, n] with even n");
    c10::cuda::CUDAGuard guard(G.device());
    G = G.contiguous();
    const int64_t batch = G.size(0); const int n = G.size(1), tpb = 256;
    auto W = torch::empty({batch, n}, G.options()); auto V = torch::empty_like(G);
    auto sweeps = torch::empty({batch}, G.options().dtype(torch::kInt32));
    const size_t small = (size_t)(n + tpb) * sizeof(double);           // cs (n doubles) + reduction scratch
    const size_t full = (size_t)2 * n * n * sizeof(double) + small;
    const size_t limit = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
    const int lds = full <= limit;
    torch::Tensor work = lds ? torch::empty({0}, G.options()) : torch::empty({batch, 2, n, n}, G.options());
    jacobi_kernel<<<batch, tpb, lds ? full : small, at::cuda::getCurrentCUDAStream()>>>(
        G.data_ptr<double>(), W.data_ptr<double>(), V.data_ptr<double>(), lds ? nullptr : work.data_ptr<double>(),
        n, lds, (int)max_sweeps, tol, sweeps.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {W, V, sweeps};
}
