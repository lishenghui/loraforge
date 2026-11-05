// SPDX-License-Identifier: Apache-2.0
// Copyright (c) Shenghui Li. Derived from lishenghui/probe 5fd6ae060d106066b306959f9a34b726a281ef4c
// tools/llm_fleet/sglang_plugin/probe_sglang_gemv/fast_decode.cu
// Arithmetic retained; host contract guards and HIP launch parsing fixed here.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
constexpr int LORAFORGE_WARP_SIZE = 64;
#define LORAFORGE_SHFL_DOWN(value, delta) __shfl_down(value, delta, 64)
#else
#include <cuda_runtime.h>
constexpr int LORAFORGE_WARP_SIZE = 32;
#define LORAFORGE_SHFL_DOWN(value, delta) __shfl_down_sync(0xffffffffu, value, delta)
#endif
constexpr int LORAFORGE_SHRINK_THREADS = 256;
constexpr int LORAFORGE_SHRINK_WARPS = LORAFORGE_SHRINK_THREADS / LORAFORGE_WARP_SIZE;
__device__ __forceinline__ float bf(uint16_t x){return __uint_as_float(uint32_t(x)<<16);}
__device__ __forceinline__ uint16_t rb(float x){uint32_t u=__float_as_uint(x);if((u&0x7f800000u)!=0x7f800000u)u+=0x7fffu+((u>>16)&1u);return uint16_t(u>>16);}
// Fixed capacity four; retained rank and routes remain live GPU metadata.
// Preserve the original lane/chunk accumulation order and BF16 rounding.
template<int K> __global__ void rank4_ab_kernel(const uint16_t* x,const uint16_t* a,const uint16_t* b,uint16_t* y,const int* routes,const int* ranks,const float* scales,int64_t xs,int64_t ys){
 int row=blockIdx.x,g=routes[row];if(g<0)return;int r=ranks[g];if(!r)return;
 int lane=threadIdx.x%64,wave=threadIdx.x/64;__shared__ float u[4];
 if(wave<r){float sum=0;const uint16_t* ap=a+(int64_t(g)*4+wave)*K;const uint16_t* xp=x+int64_t(row)*xs;
  #pragma unroll
  for(int chunk=0;chunk<K/512;chunk++){int t=lane*8+chunk*512;uint4 av=*reinterpret_cast<const uint4*>(ap+t),xv=*reinterpret_cast<const uint4*>(xp+t);auto aa=reinterpret_cast<const uint16_t*>(&av);auto xx=reinterpret_cast<const uint16_t*>(&xv);
   #pragma unroll
   for(int i=0;i<8;i++)sum+=bf(aa[i])*bf(xx[i]);}
  #pragma unroll
  for(int d=32;d>0;d>>=1)sum+=__shfl_down(sum,d,64);
  if(lane==0)u[wave]=bf(rb(sum));
 }
 __syncthreads();int n=blockIdx.y*256+threadIdx.x;
 uint2 bits=*reinterpret_cast<const uint2*>(b+(int64_t(g)*1024+n)*4);auto bp=reinterpret_cast<const uint16_t*>(&bits);float acc=0;
 #pragma unroll
 for(int k=0;k<4;k++)if(k<r)acc+=u[k]*bf(bp[k]);
 uint16_t* yp=y+int64_t(row)*ys+n;*yp=rb(bf(*yp)+bf(rb(acc*scales[g])));
}
void update(torch::Tensor x,torch::Tensor a,torch::Tensor b,torch::Tensor y,torch::Tensor routes,torch::Tensor ranks,torch::Tensor scales){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&a.scalar_type()==x.scalar_type()&&b.scalar_type()==x.scalar_type()&&y.scalar_type()==x.scalar_type());
 TORCH_CHECK(x.dim()==2&&a.dim()==3&&b.dim()==3&&y.dim()==2&&a.size(1)==4&&b.size(1)==1024&&b.size(2)==4&&y.size(1)==1024);
 TORCH_CHECK(a.is_contiguous()&&b.is_contiguous()&&x.stride(1)==1&&y.stride(1)==1&&(x.size(1)==2048||x.size(1)==3584));
 if(!x.size(0))return;c10::cuda::CUDAGuard guard(x.device());
 #define R4(K) rank4_ab_kernel<K><<<dim3(x.size(0),4),256,0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)x.data_ptr(),(uint16_t*)a.data_ptr(),(uint16_t*)b.data_ptr(),(uint16_t*)y.data_ptr(),routes.data_ptr<int>(),ranks.data_ptr<int>(),scales.data_ptr<float>(),x.stride(0),y.stride(0))
 if(x.size(1)==2048){R4(2048);}else{R4(3584);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("update",&update);}
