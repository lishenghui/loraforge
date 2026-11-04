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
// Research variants retain the attributed HIP shrink's reduction and BF16 rounding.
__global__ void shared_a_kernel(const uint16_t* x,const uint16_t* a,uint16_t* z,const int* map,const int* ranks,int K,int R,int maxr,int64_t xs){
 int row=blockIdx.x,g=map[row];if(g<0)return;int r=ranks[g];if(!r)return;
 int lane=threadIdx.x%64,wave=threadIdx.x/64,j=blockIdx.y*4+wave,sl=j/maxr,k=j%maxr;if(sl>=2||k>=r)return;
 int col=(sl?3*r:0)+k;float sum=0;const uint16_t* ap=a+(int64_t(g)*4*R+col)*K;const uint16_t* xp=x+int64_t(row)*xs;
 for(int t=lane*8;t<K;t+=512){uint4 av=*reinterpret_cast<const uint4*>(ap+t),xv=*reinterpret_cast<const uint4*>(xp+t);auto aa=reinterpret_cast<const uint16_t*>(&av);auto xx=reinterpret_cast<const uint16_t*>(&xv);
 #pragma unroll
 for(int i=0;i<8;i++)sum+=bf(aa[i])*bf(xx[i]);}
 #pragma unroll
 for(int d=32;d>0;d>>=1)sum+=__shfl_down(sum,d,64);
 if(lane==0){uint16_t v=rb(sum);if(sl)z[int64_t(row)*4*R+3*r+k]=v;else{z[int64_t(row)*4*R+k]=v;z[int64_t(row)*4*R+r+k]=v;z[int64_t(row)*4*R+2*r+k]=v;}}
}
template<int TILE> __global__ void fused_ab_kernel(const uint16_t* x,const uint16_t* a,const uint16_t* b,uint16_t* y,const int* routes,const int* ranks,const float* scales,int K,int N,int R,int64_t xs,int64_t ys){
 int row=blockIdx.x,g=routes[row];if(g<0)return;int r=ranks[g];if(!r)return;
 __shared__ float z[8];
 int lane=threadIdx.x%64,wave=threadIdx.x/64;
 for(int k=wave;k<r;k+=4){float sum=0;const uint16_t* ap=a+(int64_t(g)*R+k)*K;const uint16_t* xp=x+int64_t(row)*xs;
  for(int t=lane*8;t<K;t+=512){uint4 av=*reinterpret_cast<const uint4*>(ap+t),xv=*reinterpret_cast<const uint4*>(xp+t);auto aa=reinterpret_cast<const uint16_t*>(&av);auto xx=reinterpret_cast<const uint16_t*>(&xv);
   #pragma unroll
   for(int i=0;i<8;i++)sum+=bf(aa[i])*bf(xx[i]);}
  #pragma unroll
  for(int d=32;d>0;d>>=1)sum+=__shfl_down(sum,d,64);
  if(lane==0)z[k]=bf(rb(sum));
 }
 __syncthreads();
 float scale=scales[g];
 for(int n=blockIdx.y*TILE+threadIdx.x;n<min(N,(blockIdx.y+1)*TILE);n+=256){
  float acc=0;const uint16_t* bp=b+(int64_t(g)*N+n)*R;
  for(int k=0;k<r;k++)acc+=z[k]*bf(bp[k]);
  uint16_t* yp=y+int64_t(row)*ys+n;*yp=rb(bf(*yp)+bf(rb(acc*scale)));
 }
}
void shared_a(torch::Tensor x,torch::Tensor a,torch::Tensor z,torch::Tensor routes,torch::Tensor ranks,int64_t maxr){
 TORCH_CHECK(x.is_cuda()&&a.is_cuda()&&z.is_cuda()&&routes.is_cuda()&&ranks.is_cuda());
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&a.scalar_type()==x.scalar_type()&&z.scalar_type()==x.scalar_type());
 TORCH_CHECK(a.is_contiguous()&&z.is_contiguous()&&x.stride(1)==1&&x.size(1)%8==0&&a.size(1)%4==0&&maxr>=1&&maxr<=8);
 if(!x.size(0))return;c10::cuda::CUDAGuard guard(x.device());
 shared_a_kernel<<<dim3(x.size(0),(2*maxr+3)/4),256,0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)x.data_ptr(),(uint16_t*)a.data_ptr(),(uint16_t*)z.data_ptr(),routes.data_ptr<int>(),ranks.data_ptr<int>(),x.size(1),a.size(1)/4,maxr,x.stride(0));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void fused_ab(torch::Tensor x,torch::Tensor a,torch::Tensor b,torch::Tensor y,torch::Tensor routes,torch::Tensor ranks,torch::Tensor scales,int64_t tile){
 TORCH_CHECK(x.is_cuda()&&a.is_cuda()&&b.is_cuda()&&y.is_cuda()&&routes.is_cuda()&&ranks.is_cuda()&&scales.is_cuda());
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&a.scalar_type()==x.scalar_type()&&b.scalar_type()==x.scalar_type()&&y.scalar_type()==x.scalar_type());
 TORCH_CHECK(a.is_contiguous()&&b.is_contiguous()&&x.stride(1)==1&&x.size(1)%8==0&&y.stride(1)==1&&a.size(1)<=8&&b.size(2)==a.size(1));
 TORCH_CHECK(tile==128||tile==256||tile==512||tile==1024);
 if(!x.size(0))return;c10::cuda::CUDAGuard guard(x.device());
 #define AB(T) fused_ab_kernel<T><<<dim3(x.size(0),(y.size(1)+T-1)/T),256,0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)x.data_ptr(),(uint16_t*)a.data_ptr(),(uint16_t*)b.data_ptr(),(uint16_t*)y.data_ptr(),routes.data_ptr<int>(),ranks.data_ptr<int>(),scales.data_ptr<float>(),x.size(1),y.size(1),a.size(1),x.stride(0),y.stride(0))
 if(tile==128){AB(128);}else if(tile==256){AB(256);}else if(tile==512){AB(512);}else{AB(1024);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("shared_a",&shared_a);m.def("fused_ab",&fused_ab);}
