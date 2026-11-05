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
// Trusted-input BF16 TP1 Qwen3.5-0.8B research variants; 64-lane HIP only.
__device__ __forceinline__ float contract(const uint16_t* xp,const uint16_t* ap,int K){
 int lane=threadIdx.x%64;float sum=0;
 for(int t=lane*8;t<K;t+=512){uint4 av=*reinterpret_cast<const uint4*>(ap+t),xv=*reinterpret_cast<const uint4*>(xp+t);auto aa=reinterpret_cast<const uint16_t*>(&av);auto xx=reinterpret_cast<const uint16_t*>(&xv);
 #pragma unroll
 for(int i=0;i<8;i++)sum+=bf(aa[i])*bf(xx[i]);}
 #pragma unroll
 for(int d=32;d>0;d>>=1)sum+=__shfl_down(sum,d,64);
 return sum;
}
__global__ void wide_ab_kernel(const uint16_t* x,const uint16_t* a,const uint16_t* b,uint16_t* y,const int* routes,const int* ranks,const float* scales,int K,int N,int R,int S,int split0,int split1,int64_t xs,int64_t ys){
 int row=blockIdx.x,g=routes[row];if(g<0)return;int r=ranks[g];if(!r)return;
 int start=blockIdx.y*256,sl=start>=split0;if(S==3)sl+=start>=split1;if(S==4)sl=start/split0;
 __shared__ float u[8];int lane=threadIdx.x%64,wave=threadIdx.x/64;
 for(int k=wave;k<r;k+=4){float sum=contract(x+int64_t(row)*xs,a+(int64_t(g)*S*R+sl*r+k)*K,K);if(lane==0)u[k]=bf(rb(sum));}
 __syncthreads();int n=start+threadIdx.x;
 if(n<N){float acc=0;for(int k=0;k<r;k++)acc+=u[k]*bf(b[(int64_t(g)*N+n)*R+k]);uint16_t* yp=y+int64_t(row)*ys+n;*yp=rb(bf(*yp)+bf(rb(acc*scales[g])));}
}
__global__ void gdn_ab_unpack_kernel(const uint16_t* x,const uint16_t* aq,const uint16_t* ab,const uint16_t* bq,const uint16_t* bb,const uint16_t* qbase,const uint16_t* bbase,uint16_t* out,uint16_t* zout,uint16_t* bout,uint16_t* aout,const int* routes,const int* rq,const int* rba,const float* scales,int R,int64_t xs,int64_t qs,int64_t bs){
 int row=blockIdx.x,g=routes[row],safe=max(0,g);bool isq=blockIdx.y<32;int r=g<0?0:(isq?rq[safe]:rba[safe]);
 int lane=threadIdx.x%64,wave=threadIdx.x/64;__shared__ float u[16];
 int sl=blockIdx.y/8,parts=isq?1:2;
 const uint16_t* aa=isq?aq:ab;int slices=isq?4:2;
 for(int col=wave;col<parts*r;col+=4){int ac=isq?sl*r+col:col;float sum=contract(x+int64_t(row)*xs,aa+(int64_t(safe)*slices*R+ac)*1024,1024);if(lane==0)u[col]=bf(rb(sum));}
 __syncthreads();float scale=scales[safe];
 if(isq){int j=blockIdx.y*256+threadIdx.x;float acc=0;for(int k=0;k<r;k++)acc+=u[k]*bf(bq[(int64_t(safe)*8192+j)*R+k]);uint16_t v=rb(bf(qbase[int64_t(row)*qs+j])+bf(rb(acc*scale)));if(j<6144)out[int64_t(row)*6144+j]=v;else zout[int64_t(row)*2048+j-6144]=v;}
 else if(threadIdx.x<32){int j=threadIdx.x;float acc=0;for(int k=0;k<r;k++)acc+=u[(j/16)*r+k]*bf(bb[(int64_t(safe)*32+j)*R+k]);uint16_t v=rb(bf(bbase[int64_t(row)*bs+j])+bf(rb(acc*scale)));if(j<16)bout[int64_t(row)*16+j]=v;else aout[int64_t(row)*16+j-16]=v;}
}
void wide_ab(torch::Tensor x,torch::Tensor a,torch::Tensor b,torch::Tensor y,torch::Tensor routes,torch::Tensor ranks,torch::Tensor scales,int64_t slices,int64_t split0,int64_t split1){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&a.is_contiguous()&&b.is_contiguous()&&y.stride(1)==1&&x.stride(1)==1);
 TORCH_CHECK(slices>=2&&slices<=4&&b.size(2)<=8&&a.size(1)==slices*b.size(2)&&x.size(1)%8==0&&split0%256==0&&split1%256==0);
 if(!x.size(0))return;c10::cuda::CUDAGuard guard(x.device());
 wide_ab_kernel<<<dim3(x.size(0),(y.size(1)+255)/256),256,0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)x.data_ptr(),(uint16_t*)a.data_ptr(),(uint16_t*)b.data_ptr(),(uint16_t*)y.data_ptr(),routes.data_ptr<int>(),ranks.data_ptr<int>(),scales.data_ptr<float>(),x.size(1),y.size(1),b.size(2),slices,split0,split1,x.stride(0),y.stride(0));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gdn_ab_unpack(torch::Tensor x,torch::Tensor aq,torch::Tensor ab,torch::Tensor bq,torch::Tensor bb,torch::Tensor q,torch::Tensor ba,torch::Tensor out,torch::Tensor z,torch::Tensor bo,torch::Tensor ao,torch::Tensor routes,torch::Tensor rq,torch::Tensor rb_,torch::Tensor scales){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&x.size(1)==1024&&x.stride(1)==1&&aq.is_contiguous()&&ab.is_contiguous()&&bq.is_contiguous()&&bb.is_contiguous());
 TORCH_CHECK(q.size(1)==8192&&ba.size(1)==32&&q.stride(1)==1&&ba.stride(1)==1&&bq.size(2)<=8&&bb.size(2)==bq.size(2));
 if(!x.size(0))return;c10::cuda::CUDAGuard guard(x.device());
 gdn_ab_unpack_kernel<<<dim3(x.size(0),33),256,0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)x.data_ptr(),(uint16_t*)aq.data_ptr(),(uint16_t*)ab.data_ptr(),(uint16_t*)bq.data_ptr(),(uint16_t*)bb.data_ptr(),(uint16_t*)q.data_ptr(),(uint16_t*)ba.data_ptr(),(uint16_t*)out.data_ptr(),(uint16_t*)z.data_ptr(),(uint16_t*)bo.data_ptr(),(uint16_t*)ao.data_ptr(),routes.data_ptr<int>(),rq.data_ptr<int>(),rb_.data_ptr<int>(),scales.data_ptr<float>(),bq.size(2),x.stride(0),q.stride(0),ba.stride(0));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("wide_ab",&wide_ab);m.def("gdn_ab_unpack",&gdn_ab_unpack);}
