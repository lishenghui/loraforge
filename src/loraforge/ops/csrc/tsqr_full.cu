// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Shenghui Li
// Derived from lishenghui/probe at 5fd6ae060d106066b306959f9a34b726a281ef4c.
// Modified for owned state, device guards, and current-stream execution.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Batched TSQR, full operator (R + reflectors + Q@v). DEFAULT KERNEL.
//
// Merge nodes are TT-structured (triangle-on-triangle) AND packed:
//  * compute: column j touches only its j+2 active rows (~N^3/3 merge flops);
//  * storage: the two triangles live packed in shared memory (~N^2 floats
//    instead of the dense 2N^2), so at N=128 the merge tile drops from 131 KB
//    (1 block/SM) to ~67 KB (3 blocks/SM on GH200);
//  * inter-level R blocks and the merge reflectors travel packed through
//    global memory as well (~2x less traffic), and the Q@v application is
//    chunked over columns so large k no longer blows the shared-memory budget.
// See kernels/tsqr_dense_merge.cu for the dense-merge ablation baseline.

#define KC_CHUNK 64   // apply-phase column chunk (upper bound; shrunk to fit shared memory)

// Warp width. torch hipifies this file on ROCm builds, where AMD CDNA GPUs
// (e.g. MI250X on LUMI) execute 64-lane wavefronts; a 32-lane reduction there
// silently sums only half of each wavefront.
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
#define FT_WARP 64
#define FT_SHFL_DOWN(v, off) __shfl_down(v, off)
#else
#define FT_WARP 32
#define FT_SHFL_DOWN(v, off) __shfl_down_sync(0xffffffff, v, off)
#endif

// ---------------- packed-triangle indexing -----------------------------------
__device__ __forceinline__ int tri(int k) { return (k * (k + 1)) >> 1; }
__device__ __forceinline__ int voff(int j) { return (j * (j + 3)) >> 1; }
__device__ __forceinline__ int tri_col(int t) {
    int k = (int)((sqrtf(8.f * (float)t + 1.f) - 1.f) * 0.5f);
    while (tri(k + 1) <= t) ++k;
    while (tri(k) > t) --k;
    return k;
}

// ---------------- reductions ------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = FT_WARP / 2; off > 0; off >>= 1) v += FT_SHFL_DOWN(v, off);
    return v;
}
__device__ float block_sum(float v, float* smem) {
    int lane = threadIdx.x % FT_WARP, wid = threadIdx.x / FT_WARP;
    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + FT_WARP - 1) / FT_WARP;   // <= 32 slots in red
    v = (threadIdx.x < nw) ? smem[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    float r = smem[0];
    __syncthreads();
    return r;
}

// ---------------- dense Householder QR (leaves; input is dense) --------------
__device__ void hh_qr(float* tile, int rows, int N, float* vv, float* dots, float* red,
                      float* V, float* tau) {
    for (int j = 0; j < N; ++j) {
        float loc = 0.f;
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x) {
            float x = tile[i * N + j];
            loc += x * x;
        }
        float nrm = sqrtf(block_sum(loc, red));
        float a0 = tile[j * N + j];
        float beta = (a0 >= 0.f) ? -nrm : nrm;
        if (nrm < 1e-30f) {
            if (V) { for (int i = threadIdx.x; i < rows; i += blockDim.x) V[i * N + j] = 0.f;
                     if (threadIdx.x == 0) tau[j] = 0.f; }
            __syncthreads(); continue;
        }
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x)
            vv[i] = tile[i * N + j] - ((i == j) ? beta : 0.f);
        __syncthreads();

        float locv = 0.f;
        for (int i = j + threadIdx.x; i < rows; i += blockDim.x) locv += vv[i] * vv[i];
        float vnorm2 = block_sum(locv, red);
        if (vnorm2 < 1e-30f) {
            if (V) { for (int i = threadIdx.x; i < rows; i += blockDim.x) V[i * N + j] = 0.f;
                     if (threadIdx.x == 0) tau[j] = 0.f; }
            __syncthreads(); continue;
        }
        float t = 2.f / vnorm2;

        if (V) {
            for (int i = threadIdx.x; i < rows; i += blockDim.x)
                V[i * N + j] = (i >= j) ? vv[i] : 0.f;
            if (threadIdx.x == 0) tau[j] = t;
        }

        for (int k = j + threadIdx.x; k < N; k += blockDim.x) {
            float d = 0.f;
            for (int i = j; i < rows; ++i) d += vv[i] * tile[i * N + k];
            dots[k] = d;
        }
        __syncthreads();
        int cols = N - j, nr = rows - j;
        for (int idx = threadIdx.x; idx < nr * cols; idx += blockDim.x) {
            int i = j + idx / cols, k = j + idx % cols;
            tile[i * N + k] -= t * vv[i] * dots[k];
        }
        __syncthreads();
        if (threadIdx.x == 0) tile[j * N + j] = beta;
        for (int i = j + 1 + threadIdx.x; i < rows; i += blockDim.x) tile[i * N + j] = 0.f;
        __syncthreads();
    }
}

// ---------------- apply Q (leaves; dense reflectors), column-chunked ----------
__device__ void apply_Q(float* C, const float* V, const float* tau,
                        int rows, int N, int kc, float* dots, int cooperative) {
    for (int j = N - 1; j >= 0; --j) {
        float t = tau[j];
        if (t == 0.f) { __syncthreads(); continue; }
        if (cooperative && blockDim.x % kc == 0) {
            // All threads participate in the dot products. The original path
            // uses only kc threads and traverses every row serially per column.
            int c = threadIdx.x % kc, group = threadIdx.x / kc;
            int groups = blockDim.x / kc;
            float d = 0.f;
            for (int i = j + group; i < rows; i += groups)
                d += V[i * N + j] * C[i * kc + c];
            dots[threadIdx.x] = d;
            __syncthreads();
            if (group == 0) {
                float total = dots[c];
                for (int g = 1; g < groups; ++g) total += dots[g * kc + c];
                dots[c] = total;
            }
        } else for (int c = threadIdx.x; c < kc; c += blockDim.x) {
            float d = 0.f;
            for (int i = j; i < rows; ++i) d += V[i * N + j] * C[i * kc + c];
            dots[c] = d;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < (rows - j) * kc; idx += blockDim.x) {
            int i = j + idx / kc, c = idx % kc;
            C[i * kc + c] -= t * V[i * N + j] * dots[c];
        }
        __syncthreads();
    }
}

// ---------------- factor kernels ---------------------------------------------
// gws != nullptr: the [rows, N] tile does not fit in shared memory (N > ~230 on GH200), so it lives in a
// per-block global workspace (L2-resident) and only the small vectors stay in shared memory.
__global__ void k_leaf(const float* __restrict__ B, float* Rp, float* V, float* tau,
                       int m, int N, int P, int rows, float* gws) {
    extern __shared__ float sm[];
    float* tile = gws ? gws + (size_t)blockIdx.x * rows * N : sm;
    float* vv = gws ? sm : tile + rows * N; float* dots = vv + rows; float* red = dots + N;
    int mat = blockIdx.x / P, leaf = blockIdx.x % P;
    size_t off = ((size_t)mat * P + leaf);
    const float* src = B + (size_t)mat * m * N;
    for (int t = threadIdx.x; t < rows * N; t += blockDim.x) {
        int i = t / N;                       // zero-fill rows past m (padding)
        int gi = leaf * rows + i;
        tile[t] = (gi < m) ? src[(size_t)gi * N + (t % N)] : 0.f;
    }
    __syncthreads();
    hh_qr(tile, rows, N, vv, dots, red, V + off * (size_t)rows * N, tau + off * N);
    int T = tri(N);
    float* dst = Rp + off * (size_t)T;      // packed upper triangle out
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        int k = tri_col(t), i = t - tri(k);
        dst[t] = tile[i * N + k];
    }
}

// merge: packed triangles in, packed triangle out, packed reflectors out
__global__ void k_merge_tt(const float* __restrict__ Rin, float* Rout, float* V, float* tau,
                           int N, int pairs, int PinTotal, int PoutTotal, float* gws) {
    extern __shared__ float sm[];
    int T = tri(N);
    float* P1 = gws ? gws + (size_t)blockIdx.x * 2 * T : sm; float* P2 = P1 + T;
    float* vv = gws ? sm : P2 + T;
    float* dots = vv + (N + 1); float* red = dots + N;
    int mat = blockIdx.x / pairs, pair = blockIdx.x % pairs;
    size_t oi = ((size_t)mat * PinTotal + 2 * pair), oo = ((size_t)mat * PoutTotal + pair);
    const float* r0 = Rin + oi * (size_t)T;
    const float* r1 = Rin + (oi + 1) * (size_t)T;
    for (int t = threadIdx.x; t < T; t += blockDim.x) P1[t] = r0[t];
    for (int t = threadIdx.x; t < T; t += blockDim.x) P2[t] = r1[t];
    __syncthreads();

    size_t ov = ((size_t)mat * pairs + pair);       // V/tau exist only for merged pairs
    float* Vg = V + ov * (size_t)(2 * N) * N;       // dense [2N, N] reflectors (for WY apply)
    float* tg = tau + ov * N;
    for (int j = 0; j < N; ++j) {
        int nnz = j + 2;
        float loc = 0.f;
        for (int t = threadIdx.x; t < nnz; t += blockDim.x) {
            float x = (t == 0) ? P1[tri(j) + j] : P2[tri(j) + t - 1];
            loc += x * x;
        }
        float nrm = sqrtf(block_sum(loc, red));
        float a0 = P1[tri(j) + j];
        float beta = (a0 >= 0.f) ? -nrm : nrm;
        if (nrm < 1e-30f) {
            if (threadIdx.x == 0) tg[j] = 0.f;
            for (int t = threadIdx.x; t < 2 * N; t += blockDim.x) Vg[(size_t)t * N + j] = 0.f;
            __syncthreads(); continue;
        }
        for (int t = threadIdx.x; t < nnz; t += blockDim.x)
            vv[t] = (t == 0) ? (a0 - beta) : P2[tri(j) + t - 1];
        __syncthreads();
        float locv = 0.f;
        for (int t = threadIdx.x; t < nnz; t += blockDim.x) locv += vv[t] * vv[t];
        float vn2 = block_sum(locv, red);
        if (vn2 < 1e-30f) {
            if (threadIdx.x == 0) tg[j] = 0.f;
            for (int t = threadIdx.x; t < 2 * N; t += blockDim.x) Vg[(size_t)t * N + j] = 0.f;
            __syncthreads(); continue;
        }
        float tval = 2.f / vn2;
        for (int t = threadIdx.x; t < 2 * N; t += blockDim.x) {   // dense column j
            float val = 0.f;
            if (t == j) val = vv[0];
            else if (t >= N && t - N <= j) val = vv[t - N + 1];
            Vg[(size_t)t * N + j] = val;
        }
        if (threadIdx.x == 0) tg[j] = tval;

        for (int kc = j + 1 + threadIdx.x; kc < N; kc += blockDim.x) {
            const float* col2 = P2 + tri(kc);
            float d = vv[0] * P1[tri(kc) + j];
            for (int s = 1; s < nnz; ++s) d += vv[s] * col2[s - 1];
            dots[kc] = d;
        }
        __syncthreads();
        int cols = N - 1 - j;
        for (int t = threadIdx.x; t < nnz * cols; t += blockDim.x) {
            int s = t / cols, kc = j + 1 + t % cols;
            if (s == 0) P1[tri(kc) + j]     -= tval * vv[0] * dots[kc];
            else        P2[tri(kc) + s - 1] -= tval * vv[s] * dots[kc];
        }
        __syncthreads();
        if (threadIdx.x == 0) P1[tri(j) + j] = beta;
        __syncthreads();
    }
    float* dst = Rout + oo * (size_t)T;
    for (int t = threadIdx.x; t < T; t += blockDim.x) dst[t] = P1[t];
}

__global__ void k_unpack(const float* __restrict__ Rp, float* Rd, int N) {
    int mat = blockIdx.x;
    int T = tri(N);
    for (int t = threadIdx.x; t < N * N; t += blockDim.x) {
        int i = t / N, k = t % N;
        Rd[(size_t)mat * N * N + t] = (i <= k) ? Rp[(size_t)mat * T + tri(k) + i] : 0.f;
    }
}

// ---------------- apply kernels (column-chunked) ------------------------------
__global__ void k_apply_merge_tt(const float* __restrict__ Wout, float* Win,
                                 const float* V, const float* tau,
                                 int N, int pairs, int PoutTotal, int PinTotal,
                                 int rank, int chunks, int kcs) {
    extern __shared__ float sm[];
    int rows = 2 * N;
    int nodeIdx = blockIdx.x / chunks, ch = blockIdx.x % chunks;
    int mat = nodeIdx / pairs, node = nodeIdx % pairs;
    int c0 = ch * kcs, kc = min(kcs, rank - c0);
    float* C = sm; float* dots = C + (size_t)rows * kcs;
    size_t o = ((size_t)mat * PoutTotal + node);
    size_t ov = ((size_t)mat * pairs + node);

    for (int t = threadIdx.x; t < N * kc; t += blockDim.x) {
        int i = t / kc, c = t % kc;
        C[i * kc + c] = Wout[o * (size_t)N * rank + (size_t)i * rank + (c0 + c)];
    }
    for (int t = N * kc + threadIdx.x; t < rows * kc; t += blockDim.x) C[t] = 0.f;
    __syncthreads();

    const float* Vg = V + ov * (size_t)(2 * N) * N;   // dense [2N, N]
    const float* tg = tau + ov * N;
    for (int j = N - 1; j >= 0; --j) {
        float t = tg[j];
        if (t == 0.f) { __syncthreads(); continue; }
        int nnz = j + 2;
        for (int c = threadIdx.x; c < kc; c += blockDim.x) {
            float d = Vg[(size_t)j * N + j] * C[j * kc + c];
            for (int s = 1; s < nnz; ++s)
                d += Vg[(size_t)(N + s - 1) * N + j] * C[(N + s - 1) * kc + c];
            dots[c] = d;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < nnz * kc; idx += blockDim.x) {
            int s = idx / kc, c = idx % kc;
            int i = (s == 0) ? j : (N + s - 1);
            float vij = (s == 0) ? Vg[(size_t)j * N + j] : Vg[(size_t)(N + s - 1) * N + j];
            C[i * kc + c] -= t * vij * dots[c];
        }
        __syncthreads();
    }
    size_t oi = ((size_t)mat * PinTotal + 2 * node);
    for (int t = threadIdx.x; t < N * kc; t += blockDim.x) {
        int i = t / kc, c = t % kc;
        Win[oi * (size_t)N * rank + (size_t)i * rank + (c0 + c)] = C[i * kc + c];
        Win[(oi + 1) * (size_t)N * rank + (size_t)i * rank + (c0 + c)] = C[(N + i) * kc + c];
    }
}

__global__ void k_apply_leaf(const float* __restrict__ W, float* Out,
                             const float* V, const float* tau,
                             int m, int N, int P, int rows, int rank, int chunks, int kcs, int cooperative) {
    extern __shared__ float sm[];
    int nodeIdx = blockIdx.x / chunks, ch = blockIdx.x % chunks;
    int mat = nodeIdx / P, leaf = nodeIdx % P;
    int c0 = ch * kcs, kc = min(kcs, rank - c0);
    float* C = sm; float* dots = C + (size_t)rows * kcs;
    size_t o = ((size_t)mat * P + leaf);

    for (int t = threadIdx.x; t < N * kc; t += blockDim.x) {
        int i = t / kc, c = t % kc;
        C[i * kc + c] = W[o * (size_t)N * rank + (size_t)i * rank + (c0 + c)];
    }
    for (int t = N * kc + threadIdx.x; t < rows * kc; t += blockDim.x) C[t] = 0.f;
    __syncthreads();
    apply_Q(C, V + o * (size_t)rows * N, tau + o * N, rows, N, kc, dots, cooperative);
    float* dst = Out + (size_t)mat * m * rank;
    for (int t = threadIdx.x; t < rows * kc; t += blockDim.x) {
        int i = t / kc, c = t % kc;
        int gi = leaf * rows + i;
        if (gi < m) dst[(size_t)gi * rank + (c0 + c)] = C[i * kc + c];
    }
}

// ---------------- host API ---------------------------------------------------
#define WY_NB 32   // WY panel width for the blocked apply

struct Handle {
    torch::Tensor Rp, Vleaf, tleaf, Tleaf;
    std::vector<torch::Tensor> Vm, tm, Tm;
    std::vector<int> inCnt, outCnt;   // node counts per merge level (bye-aware)
    int m, N, P, rows;
};


// Largest dynamic shared-memory request a block may make (opt-in limit on
// CUDA; 64 KB LDS per workgroup on AMD CDNA).
static size_t max_dyn_smem() {
    auto* p = at::cuda::getCurrentDeviceProperties();
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
    return p->sharedMemPerBlock;
#else
    return p->sharedMemPerBlockOptin;
#endif
}
int64_t tsqr_shared_memory_limit(torch::Tensor x) {
    c10::cuda::CUDAGuard guard(x.device());
    return static_cast<int64_t>(max_dyn_smem());
}
template <typename K> static void allow_smem(K kernel, size_t bytes, const char* what) {
    TORCH_CHECK(bytes <= max_dyn_smem(), what, " needs ", bytes, " B shared memory, device allows ",
                max_dyn_smem(), " B; lower rows_per_leaf");
#if !(defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__))
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)max_dyn_smem());
#endif
}
static void launch_check(const char* what) {
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, what, " launch failed: ", cudaGetErrorString(e));
}
// apply-phase column chunk: as wide as KC_CHUNK allows, narrowed to fit shared memory
static int chunk_width(int rows, int rank) {
    int kcs = std::min(KC_CHUNK, rank);
    while (kcs > 1 && ((size_t)rows * kcs + kcs + 32) * sizeof(float) > max_dyn_smem()) kcs = (kcs + 1) / 2;
    return kcs;
}

// T factors for panels of WY_NB reflectors: V [L, rows, N], tau [L, N]
// returns T [L, npan, nb, nb] with the last panel zero-padded.
static torch::Tensor build_T(torch::Tensor V, torch::Tensor tau, int N) {
    // Closed form: T^{-1} = diag(1/tau) + triu(Y, 1) with Y = V^T V, so each
    // panel is one batched bmm + one batched triangular solve (no column loop).
    int nb = WY_NB, npan = (N + nb - 1) / nb;
    auto L = V.size(0);
    auto Tt = torch::zeros({L, npan, nb, nb}, V.options());
    auto eye = at::eye(nb, V.options());
    for (int p = 0; p < npan; ++p) {
        int j0 = p * nb, w = std::min(nb, N - j0);
        auto Vp = V.narrow(2, j0, w);
        auto Y = at::bmm(Vp.transpose(1, 2), Vp);                  // [L, w, w]
        auto tp = tau.narrow(1, j0, w);                            // [L, w]
        auto invd = at::where(tp.abs() < 1e-30, at::full_like(tp, 1e30), tp.reciprocal());
        auto A = at::triu(Y, 1) + at::diag_embed(invd);            // [L, w, w] upper
        auto Tp = at::linalg_solve_triangular(A, eye.narrow(0, 0, w).narrow(1, 0, w)
                                                   .expand({L, w, w}),
                                              /*upper=*/true, /*left=*/true,
                                              /*unitriangular=*/false);
        Tt.select(1, p).narrow(1, 0, w).narrow(2, 0, w).copy_(Tp);
    }
    return Tt;
}

// blocked WY application of Q = prod_j H_j to C [L, rows, k], panels in reverse
static void wy_apply(torch::Tensor C, torch::Tensor V, torch::Tensor Tt, int N) {
    int nb = WY_NB, npan = (N + nb - 1) / nb;
    for (int p = npan - 1; p >= 0; --p) {
        int j0 = p * nb, w = std::min(nb, N - j0);
        auto Vp = V.narrow(2, j0, w);                        // [L, rows, w]
        auto W = at::bmm(Vp.transpose(1, 2), C);             // [L, w, k]
        W = at::bmm(Tt.select(1, p).narrow(1, 0, w).narrow(2, 0, w), W);
        C.sub_(at::bmm(Vp, W));
    }
}

std::vector<torch::Tensor> tsqr_factor(torch::Tensor B, int64_t rows_per_leaf, int64_t tpb) {
    c10::cuda::CUDAGuard guard(B.device());
    Handle h;
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kFloat32 && B.dim() == 3);
    B = B.contiguous();
    int M = B.size(0), m = B.size(1), N = B.size(2);
    TORCH_CHECK(rows_per_leaf >= N, "rows_per_leaf must be >= N");
    int P = (int)((m + rows_per_leaf - 1) / rows_per_leaf);   // last leaf zero-padded
    auto o = B.options();
    int T = (N * (N + 1)) / 2, VT = (N * (N + 3)) / 2;

    auto Rp = torch::empty({M, P, T}, o);
    auto Vl = torch::empty({M, P, (int)rows_per_leaf, N}, o);
    auto tl = torch::empty({M, P, N}, o);
    size_t small = (rows_per_leaf + N + 32) * sizeof(float);
    size_t sh = rows_per_leaf * N * sizeof(float) + small;
    torch::Tensor lws;                       // global tile workspace when the tile exceeds shared memory
    if (sh > max_dyn_smem()) { lws = torch::empty({(int64_t)M * P, rows_per_leaf * N}, o); sh = small; }
    allow_smem(k_leaf, sh, "k_leaf");
    k_leaf<<<M * P, tpb, sh, at::cuda::getCurrentCUDAStream()>>>(B.data_ptr<float>(), Rp.data_ptr<float>(),
                               Vl.data_ptr<float>(), tl.data_ptr<float>(), m, N, P, rows_per_leaf,
                               lws.defined() ? lws.data_ptr<float>() : nullptr);
    launch_check("k_leaf");

    h.Vm.clear(); h.tm.clear(); h.Tm.clear(); h.inCnt.clear(); h.outCnt.clear();
    size_t msmall = ((N + 1) + N + 32) * sizeof(float);
    size_t shm = 2 * T * sizeof(float) + msmall;
    bool mglobal = shm > max_dyn_smem();
    if (mglobal) shm = msmall;
    allow_smem(k_merge_tt, shm, "k_merge_tt");
    auto cur = Rp; int Pin = P;
    while (Pin > 1) {
        int pairs = Pin / 2, bye = Pin & 1, Pout = pairs + bye;
        auto nxt = torch::empty({M, Pout, T}, o);
        auto Vm = torch::empty({M, pairs, 2 * N, N}, o);
        auto tm = torch::empty({M, pairs, N}, o);
        torch::Tensor mws;
        if (mglobal) mws = torch::empty({(int64_t)M * pairs, 2 * T}, o);
        k_merge_tt<<<M * pairs, tpb, shm, at::cuda::getCurrentCUDAStream()>>>(cur.data_ptr<float>(), nxt.data_ptr<float>(),
                                            Vm.data_ptr<float>(), tm.data_ptr<float>(),
                                            N, pairs, Pin, Pout,
                                            mglobal ? mws.data_ptr<float>() : nullptr);
        launch_check("k_merge_tt");
        if (bye)   // odd node passes through untouched
            nxt.select(1, Pout - 1).copy_(cur.select(1, Pin - 1));
        h.Vm.push_back(Vm); h.tm.push_back(tm);
        h.inCnt.push_back(Pin); h.outCnt.push_back(Pout);
        cur = nxt; Pin = Pout;
    }
    h.Rp = cur; h.Vleaf = Vl; h.tleaf = tl;
    h.Tleaf = torch::empty({0}, o);          // T factors are populated before publishing the immutable state
    h.m = m; h.N = N; h.P = P; h.rows = rows_per_leaf;

    auto Rd = torch::empty({M, N, N}, o);
    k_unpack<<<M, tpb, 0, at::cuda::getCurrentCUDAStream()>>>(cur.data_ptr<float>(), Rd.data_ptr<float>(), N);
    launch_check("k_unpack");
    int rows = rows_per_leaf;
    if (N >= 160) {
        h.Tm.clear();
        for (size_t L = 0; L < h.Vm.size(); ++L) {
            int pairs = h.Vm[L].size(1);
            h.Tm.push_back(build_T(h.Vm[L].view({(int64_t)M * pairs, 2 * N, N}),
                                     h.tm[L].view({(int64_t)M * pairs, N}), N));
        }
        h.Tleaf = build_T(h.Vleaf.view({(int64_t)M * P, rows, N}),
                            h.tleaf.view({(int64_t)M * P, N}), N);
    }
    std::vector<torch::Tensor> state{Rd, Vl, tl, h.Tleaf};
    for (size_t L = 0; L < h.Vm.size(); ++L) {
        state.push_back(h.Vm[L]); state.push_back(h.tm[L]);
        state.push_back(N >= 160 ? h.Tm[L] : torch::empty({0}, o));
    }
    return state;
}

torch::Tensor tsqr_applyQ(torch::Tensor v, const std::vector<torch::Tensor>& state,
                          int64_t m_original, int64_t tpb, bool cooperative) {
    TORCH_CHECK(state.size() >= 4 && (state.size() - 4) % 3 == 0, "invalid TSQR state");
    TORCH_CHECK(v.is_cuda() && v.scalar_type() == torch::kFloat32 && v.dim() == 3,
                "expected GPU fp32 [batch, n, k]");
    TORCH_CHECK(v.device() == state[0].device(), "apply input must match factor device");
    c10::cuda::CUDAGuard guard(v.device());
    Handle h;
    h.Vleaf = state[1]; h.tleaf = state[2]; h.Tleaf = state[3];
    h.P = h.Vleaf.size(1); h.rows = h.Vleaf.size(2); h.N = h.Vleaf.size(3); h.m = m_original;
    TORCH_CHECK(v.size(0) == state[0].size(0) && v.size(1) == h.N && v.size(2) > 0,
                "apply input shape does not match factorization");
    int count = h.P;
    for (size_t i = 4; i < state.size(); i += 3) {
        h.Vm.push_back(state[i]); h.tm.push_back(state[i+1]); h.Tm.push_back(state[i+2]);
        h.inCnt.push_back(count); count = (count + 1) / 2; h.outCnt.push_back(count);
    }
    v = v.contiguous();
    int M = v.size(0), N = h.N, rank = v.size(2), P = h.P, rows = h.rows, m = h.m;
    auto o = v.options();

    if (N < 160) {   // small N: the fused column-wise kernels win (low launch cost)
        auto w = v.reshape({M, 1, N, rank}).contiguous();
        int kcm = chunk_width(2 * N, rank), chm = (rank + kcm - 1) / kcm;
        size_t shm = ((size_t)2 * N * kcm + kcm + 32) * sizeof(float);
        allow_smem(k_apply_merge_tt, shm, "k_apply_merge_tt");
        for (int L = (int)h.Vm.size() - 1; L >= 0; --L) {
            int pairs = h.Vm[L].size(1);
            int Pin = h.inCnt[L], Pout = h.outCnt[L];
            auto win = torch::empty({M, Pin, N, rank}, o);
            k_apply_merge_tt<<<M * pairs * chm, tpb, shm, at::cuda::getCurrentCUDAStream()>>>(
                w.data_ptr<float>(), win.data_ptr<float>(),
                h.Vm[L].data_ptr<float>(), h.tm[L].data_ptr<float>(),
                N, pairs, Pout, Pin, rank, chm, kcm);
            launch_check("k_apply_merge_tt");
            if (Pin & 1)
                win.select(1, Pin - 1).copy_(w.select(1, Pout - 1));
            w = win;
        }
        auto out = torch::empty({M, m, rank}, o);
        int kcl = chunk_width(rows, rank), chl = (rank + kcl - 1) / kcl;
        size_t shl = ((size_t)rows * kcl + (cooperative ? tpb : kcl) + 32) * sizeof(float);
        allow_smem(k_apply_leaf, shl, "k_apply_leaf");
        k_apply_leaf<<<M * P * chl, tpb, shl, at::cuda::getCurrentCUDAStream()>>>(
            w.data_ptr<float>(), out.data_ptr<float>(),
            h.Vleaf.data_ptr<float>(), h.tleaf.data_ptr<float>(),
            m, N, P, rows, rank, chl, kcl, cooperative);
        launch_check("k_apply_leaf");
        return out;
    }

    // large N: blocked WY apply on cuBLAS batched GEMMs
    auto w = v.reshape({M, 1, N, rank}).contiguous();
    for (int L = (int)h.Vm.size() - 1; L >= 0; --L) {
        int pairs = h.Vm[L].size(1);
        int Pin = h.inCnt[L], Pout = h.outCnt[L];
        auto C = torch::zeros({(int64_t)M * pairs, 2 * N, rank}, o);
        C.narrow(1, 0, N).copy_(
            w.narrow(1, 0, pairs).reshape({(int64_t)M * pairs, N, rank}));
        wy_apply(C, h.Vm[L].view({(int64_t)M * pairs, 2 * N, N}), h.Tm[L], N);
        auto win = torch::empty({M, Pin, N, rank}, o);
        auto Cv = C.view({M, pairs, 2, N, rank});
        win.narrow(1, 0, 2 * pairs).view({M, pairs, 2, N, rank}).copy_(Cv);
        if (Pin & 1)
            win.select(1, Pin - 1).copy_(w.select(1, Pout - 1));
        w = win;
    }
    auto C = torch::zeros({(int64_t)M * P, rows, rank}, o);
    C.narrow(1, 0, N).copy_(w.reshape({(int64_t)M * P, N, rank}));
    wy_apply(C, h.Vleaf.view({(int64_t)M * P, rows, N}), h.Tleaf, N);
    return C.view({M, (int64_t)P * rows, rank}).narrow(1, 0, m).contiguous();
}
