// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Shenghui Li
// Derived from lishenghui/probe at 5fd6ae060d106066b306959f9a34b726a281ef4c.
// Modified for owned state, device guards, and current-stream execution.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
// FlashTSQR wide-input path: leaf and merge Householder tiles in global memory.
// One workgroup owns a tile; only a reflector vector and dot scratch use LDS.
// This removes the O(N^2) LDS bound. No generic QR backend is called.
// CUDA defaults to 48 KiB per block. Larger dynamic scratch needs opt-in;
// ROCm exposes its workgroup limit directly and has no CUDA opt-in API.
template <typename K> static void allow_wide_smem(K kernel, size_t bytes) {
 auto* props=at::cuda::getCurrentDeviceProperties();
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
 const size_t limit=props->sharedMemPerBlock;
#else
 const size_t limit=props->sharedMemPerBlockOptin;
#endif
 TORCH_CHECK(bytes<=limit,"wide scratch needs ",bytes," B shared memory, device allows ",limit," B");
#if !(defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__))
 // Always use the same ceiling so concurrent calls cannot lower the attribute
 // between another caller's validation and launch.
 C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)limit));
#endif
}
__device__ float reduce_sum(float v,float* s){
 int t=threadIdx.x;s[t]=v;__syncthreads();
 for(int d=blockDim.x/2;d;d/=2){if(t<d)s[t]+=s[t+d];__syncthreads();}
 return s[0];
}
__global__ void factor_kernel(float* X,float* V,float* tau,int rows,int n){
 extern __shared__ float sm[];float* vv=sm;float* dots=vv+rows;float* red=dots+n;
 int t=threadIdx.x;size_t off=(size_t)blockIdx.x*rows*n;X+=off;V+=off;tau+=(size_t)blockIdx.x*n;
 for(int j=0;j<n;j++){
  float sum=0;for(int i=j+t;i<rows;i+=blockDim.x){float x=X[i*n+j];sum+=x*x;}
  float norm=sqrtf(reduce_sum(sum,red));float beta=-copysignf(norm,X[j*n+j]);
  for(int i=t;i<rows;i+=blockDim.x)vv[i]=i<j?0.f:X[i*n+j]-(i==j?beta:0.f);
  __syncthreads();sum=0;for(int i=j+t;i<rows;i+=blockDim.x)sum+=vv[i]*vv[i];
  float norm2=reduce_sum(sum,red);float coef=norm2>1e-30f?2.f/norm2:0.f;
  if(t==0)tau[j]=coef;
  for(int i=t;i<rows;i+=blockDim.x)V[i*n+j]=vv[i];
  for(int k=j+t;k<n;k+=blockDim.x){float d=0;for(int i=j;i<rows;i++)d+=vv[i]*X[i*n+k];dots[k]=coef*d;}
  __syncthreads();
  for(int idx=t;idx<(rows-j)*(n-j);idx+=blockDim.x){int i=j+idx/(n-j),k=j+idx%(n-j);X[i*n+k]-=vv[i]*dots[k];}
  __syncthreads();if(t==0)X[j*n+j]=beta;
  for(int i=j+1+t;i<rows;i+=blockDim.x)X[i*n+j]=0;
  __syncthreads();
 }
}
__global__ void apply_kernel(const float* V,const float* tau,float* C,int rows,int n,int k){
 extern __shared__ float sm[];float* vv=sm;float* dots=sm+rows;int t=threadIdx.x;
 V+=(size_t)blockIdx.x*rows*n;tau+=(size_t)blockIdx.x*n;C+=(size_t)blockIdx.x*rows*k;
 for(int j=n-1;j>=0;j--){
  for(int i=t;i<rows;i+=blockDim.x)vv[i]=V[i*n+j];__syncthreads();
  for(int col=t;col<k;col+=blockDim.x){float d=0;for(int i=j;i<rows;i++)d+=vv[i]*C[i*k+col];dots[col]=tau[j]*d;}
  __syncthreads();for(int idx=t;idx<(rows-j)*k;idx+=blockDim.x){int i=j+idx/k,col=idx%k;C[i*k+col]-=vv[i]*dots[col];}__syncthreads();
 }
}
std::vector<torch::Tensor> wide_factor(torch::Tensor x){
 TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat32&&x.dim()==3&&x.size(1)>=x.size(2));
 c10::cuda::CUDAGuard guard(x.device());
 x=x.contiguous().clone();int rows=x.size(1),n=x.size(2);auto v=torch::empty_like(x);auto tau=torch::empty({x.size(0),n},x.options());
 const size_t scratch=((size_t)rows+n+256)*sizeof(float);
 allow_wide_smem(factor_kernel,scratch);
 factor_kernel<<<x.size(0),256,scratch,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),v.data_ptr<float>(),tau.data_ptr<float>(),rows,n);C10_CUDA_KERNEL_LAUNCH_CHECK();
 return {x.narrow(1,0,n).contiguous(),v,tau};
}
torch::Tensor wide_apply(torch::Tensor v,torch::Tensor tau,torch::Tensor c){
 TORCH_CHECK(v.is_cuda()&&tau.is_cuda()&&c.is_cuda()&&v.scalar_type()==torch::kFloat32&&c.scalar_type()==torch::kFloat32);
 TORCH_CHECK(v.device()==tau.device()&&v.device()==c.device(),"wide apply device mismatch");
 TORCH_CHECK(tau.scalar_type()==torch::kFloat32,"expected fp32 tau");
 c10::cuda::CUDAGuard guard(c.device());
 v=v.contiguous();tau=tau.contiguous();c=c.contiguous().clone();int rows=v.size(1),n=v.size(2),k=c.size(2);
 TORCH_CHECK(c.size(0)==v.size(0)&&c.size(1)==rows,"wide apply shape mismatch");
 const size_t scratch=((size_t)rows+k)*sizeof(float);
 allow_wide_smem(apply_kernel,scratch);
 apply_kernel<<<v.size(0),256,scratch,at::cuda::getCurrentCUDAStream()>>>(v.data_ptr<float>(),tau.data_ptr<float>(),c.data_ptr<float>(),rows,n,k);C10_CUDA_KERNEL_LAUNCH_CHECK();return c;
}
