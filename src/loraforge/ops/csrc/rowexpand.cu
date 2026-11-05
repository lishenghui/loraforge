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
__global__ void shrink_kernel(const uint16_t* x,const uint16_t* a,uint16_t* z,const int* map,const int* ranks,int K,int S,int R,int maxr,int64_t xs){
 int row=blockIdx.x,g=map[row];if(g<0)return;int r=ranks[g];if(!r)return;
 int lane=threadIdx.x%LORAFORGE_WARP_SIZE,wave=threadIdx.x/LORAFORGE_WARP_SIZE,j=blockIdx.y*LORAFORGE_SHRINK_WARPS+wave,sl=j/maxr,k=j%maxr;if(sl>=S||k>=r)return;
 int col=sl*r+k;float sum=0;const uint16_t* ap=a+(int64_t(g)*S*R+col)*K;const uint16_t* xp=x+int64_t(row)*xs;
 for(int t=lane*8;t<K;t+=LORAFORGE_WARP_SIZE*8){uint4 av=*reinterpret_cast<const uint4*>(ap+t),xv=*reinterpret_cast<const uint4*>(xp+t);auto aa=reinterpret_cast<const uint16_t*>(&av);auto xx=reinterpret_cast<const uint16_t*>(&xv);
 #pragma unroll
 for(int i=0;i<8;i++)sum+=bf(aa[i])*bf(xx[i]);}
 #pragma unroll
 for(int d=LORAFORGE_WARP_SIZE/2;d>0;d>>=1)sum+=LORAFORGE_SHFL_DOWN(sum,d);
 if(lane==0)z[int64_t(row)*S*R+col]=rb(sum);
}
template<int BR, int MAX_S> __device__ __forceinline__ void expand_tile(const uint16_t* z,const uint16_t* bp,uint16_t* y,int N,int S,int R,int64_t ys,int n,int T,int r,const int* rows,float* zs,const int* offs,float scale){
 for(int i=threadIdx.x;i<T*S*BR;i+=blockDim.x){int t=i/(S*BR),rem=i%(S*BR),sl=rem/BR,k=rem%BR;zs[(t*MAX_S+sl)*64+k]=k<r?bf(z[int64_t(rows[t])*S*R+sl*r+k]):0.f;}__syncthreads();
 if(n<N){int sl=0;for(int i=1;i<S;i++)if(n>=offs[i])sl=i;float b[BR];bp+=int64_t(n)*BR;
 if constexpr(BR>=8){
  #pragma unroll
  for(int q=0;q<BR;q+=8){uint4 v=*reinterpret_cast<const uint4*>(bp+q);auto vv=reinterpret_cast<const uint16_t*>(&v);
   #pragma unroll
   for(int j=0;j<8;j++)b[q+j]=bf(vv[j]);}
 }else if constexpr(BR==4){uint2 v=*reinterpret_cast<const uint2*>(bp);auto vv=reinterpret_cast<const uint16_t*>(&v);
  #pragma unroll
  for(int j=0;j<4;j++)b[j]=bf(vv[j]);
 }else if constexpr(BR==2){uint32_t v=*reinterpret_cast<const uint32_t*>(bp);auto vv=reinterpret_cast<const uint16_t*>(&v);b[0]=bf(vv[0]);b[1]=bf(vv[1]);}else{b[0]=bf(bp[0]);}
 for(int t=0;t<T;t++){float acc=0;
  #pragma unroll
  for(int k=0;k<BR;k++)acc+=zs[(t*MAX_S+sl)*64+k]*b[k];
  uint16_t* yp=y+int64_t(rows[t])*ys+n;*yp=rb(bf(*yp)+bf(rb(acc*scale)));}
 }__syncthreads();
}
template<int MAX_S> __global__ void expand_kernel(const uint16_t* z,const uint16_t* B,const int64_t* boff,const int* bucket,uint16_t* y,const int* seg,const int* wi,const int* perm,const int* ranks,const float* scales,const int* offs,const int* nseg,int N,int S,int R,int64_t ys){
 __shared__ int rows[16];__shared__ float zs[16*MAX_S*64];int ncb=(N+blockDim.x-1)/blockDim.x,items=(*nseg)*ncb;
 for(int item=blockIdx.x;item<items;item+=gridDim.x){int sg=item/ncb,cb=item%ncb,st=seg[sg],en=seg[sg+1],T=en-st,g=wi[sg];if(g<0)continue;int r=ranks[g];if(T<=0||r<=0)continue;
  if(T>16||r>64)return;
  if(threadIdx.x<16)rows[threadIdx.x]=threadIdx.x<T?perm[st+threadIdx.x]:0;__syncthreads();int n=cb*blockDim.x+threadIdx.x,br=bucket[g];const uint16_t* bp=B+boff[g];float scale=scales[g];
  if(br==1)expand_tile<1,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==2)expand_tile<2,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==4)expand_tile<4,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==8)expand_tile<8,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==16)expand_tile<16,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==32)expand_tile<32,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==64)expand_tile<64,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
 }
}
// Split only B work. Native A keeps its original segments and matrix tiles.
// A virtual subsegment owns disjoint output rows, with the same rank reduction.
template<int MAX_S, int ROW_TILE> __global__ void expand_tiled_kernel(const uint16_t* z,const uint16_t* B,const int64_t* boff,const int* bucket,uint16_t* y,const int* seg,const int* wi,const int* perm,const int* ranks,const float* scales,const int* offs,const int* nseg,int N,int S,int R,int64_t ys){
 constexpr int PARTS=16/ROW_TILE;
 __shared__ int rows[ROW_TILE];__shared__ float zs[ROW_TILE*MAX_S*64];
 int ncb=(N+blockDim.x-1)/blockDim.x,items=(*nseg)*PARTS*ncb;
 for(int item=blockIdx.x;item<items;item+=gridDim.x){
  int part=item/ncb,sg=part/PARTS,cb=item%ncb,start=seg[sg],end=seg[sg+1];
  if(end-start>16)return;
  int st=start+(part%PARTS)*ROW_TILE,en=min(end,st+ROW_TILE),T=en-st,g=wi[sg];
  if(T<=0||g<0)continue;int r=ranks[g];if(r<=0)continue;if(r>64)return;
  if(threadIdx.x<ROW_TILE)rows[threadIdx.x]=threadIdx.x<T?perm[st+threadIdx.x]:0;__syncthreads();
  int n=cb*blockDim.x+threadIdx.x,br=bucket[g];const uint16_t* bp=B+boff[g];float scale=scales[g];
  if(br==1)expand_tile<1,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==2)expand_tile<2,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==4)expand_tile<4,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==8)expand_tile<8,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==16)expand_tile<16,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==32)expand_tile<32,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
  else if(br==64)expand_tile<64,MAX_S>(z,bp,y,N,S,R,ys,n,T,r,rows,zs,offs,scale);
 }
}
void shrink(torch::Tensor x,torch::Tensor a,torch::Tensor z,torch::Tensor map,torch::Tensor ranks,int64_t slices,int64_t maxr){
 TORCH_CHECK(x.is_cuda()&&a.is_cuda()&&z.is_cuda()&&map.is_cuda()&&ranks.is_cuda());TORCH_CHECK(x.scalar_type()==at::kBFloat16&&a.scalar_type()==at::kBFloat16&&z.scalar_type()==at::kBFloat16);TORCH_CHECK(x.stride(1)==1&&x.stride(0)%8==0&&x.size(1)%8==0&&a.is_contiguous()&&z.is_contiguous());TORCH_CHECK(map.numel()>=x.size(0)&&map.scalar_type()==at::kInt&&ranks.scalar_type()==at::kInt);TORCH_CHECK(slices>=1&&slices<=4&&maxr>=1&&maxr<=64&&a.size(1)%slices==0&&a.size(2)==x.size(1)&&z.size(1)==a.size(1));
 TORCH_CHECK(x.dim()==2 && a.dim()==3 && z.dim()==2 && z.size(0)==x.size(0));
 TORCH_CHECK(map.is_contiguous() && ranks.is_contiguous() && ranks.numel()==a.size(0));
 for (const auto& t : {a,z,map,ranks}) TORCH_CHECK(t.device()==x.device(), "all tensors must share a device");
 TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr())%16==0 && reinterpret_cast<uintptr_t>(a.data_ptr())%16==0, "unaligned shrink operands");
 if(x.size(0)==0)return;
 c10::cuda::CUDAGuard guard(x.device());
 shrink_kernel<<<dim3(x.size(0),(slices*maxr+LORAFORGE_SHRINK_WARPS-1)/LORAFORGE_SHRINK_WARPS),dim3(LORAFORGE_SHRINK_THREADS),0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)x.data_ptr(),(uint16_t*)a.data_ptr(),(uint16_t*)z.data_ptr(),map.data_ptr<int>(),ranks.data_ptr<int>(),(int)x.size(1),(int)slices,(int)(a.size(1)/slices),(int)maxr,x.stride(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void expand(torch::Tensor z,torch::Tensor B,torch::Tensor boff,torch::Tensor bucket,torch::Tensor y,torch::Tensor seg,torch::Tensor wi,torch::Tensor perm,torch::Tensor ranks,torch::Tensor scales,torch::Tensor offs,torch::Tensor nseg,int64_t N,int64_t R,int64_t nblocks,int64_t threads,int64_t row_tile){
 TORCH_CHECK(z.is_cuda()&&B.is_cuda()&&y.is_cuda()&&boff.is_cuda()&&bucket.is_cuda()&&seg.is_cuda()&&wi.is_cuda()&&perm.is_cuda()&&ranks.is_cuda()&&scales.is_cuda()&&offs.is_cuda()&&nseg.is_cuda());TORCH_CHECK(z.scalar_type()==at::kBFloat16&&B.scalar_type()==at::kBFloat16&&y.scalar_type()==at::kBFloat16&&scales.scalar_type()==at::kFloat);TORCH_CHECK(z.is_contiguous()&&B.is_contiguous()&&y.stride(1)==1&&offs.numel()>=2&&offs.numel()<=5&&nblocks>0&&(threads==128||threads==256));TORCH_CHECK(ranks.scalar_type()==at::kInt&&bucket.scalar_type()==at::kInt&&seg.scalar_type()==at::kInt&&wi.scalar_type()==at::kInt&&perm.scalar_type()==at::kInt&&boff.scalar_type()==at::kLong&&nseg.scalar_type()==at::kInt);
 TORCH_CHECK(z.dim()==2 && B.dim()==1 && y.dim()==2 && N>0 && R>0);
 TORCH_CHECK(y.size(0)==z.size(0) && y.size(1)==N && z.size(1)==R*(offs.numel()-1));
 TORCH_CHECK(offs.scalar_type()==at::kInt && nseg.numel()==1);
 TORCH_CHECK(bucket.numel()==ranks.numel() && boff.numel()==ranks.numel() && scales.numel()==ranks.numel());
 TORCH_CHECK(seg.numel()>=wi.numel()+1 && perm.numel()>=z.size(0));
 TORCH_CHECK(y.stride(0)>=N && reinterpret_cast<uintptr_t>(B.data_ptr())%16==0);
 for (const auto& t : {B,boff,bucket,y,seg,wi,perm,ranks,scales,offs,nseg}) TORCH_CHECK(t.device()==z.device(), "all tensors must share a device");
 for (const auto& t : {boff,bucket,seg,wi,perm,ranks,scales,offs,nseg}) TORCH_CHECK(t.is_contiguous(), "metadata must be contiguous");
 TORCH_CHECK(row_tile==2||row_tile==4||row_tile==8||row_tile==16, "row_tile must be 2, 4, 8, or 16");
 if(z.size(0)==0)return;
 c10::cuda::CUDAGuard guard(z.device());
 if(row_tile!=16){
#define LF_TILED(S,T) expand_tiled_kernel<S,T><<<dim3(nblocks),dim3(threads),0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)z.data_ptr(),(uint16_t*)B.data_ptr(),boff.data_ptr<int64_t>(),bucket.data_ptr<int>(),(uint16_t*)y.data_ptr(),seg.data_ptr<int>(),wi.data_ptr<int>(),perm.data_ptr<int>(),ranks.data_ptr<int>(),scales.data_ptr<float>(),offs.data_ptr<int>(),nseg.data_ptr<int>(),(int)N,(int)(offs.numel()-1),(int)R,y.stride(0))
  if(offs.numel()==5){
   if(row_tile==2){LF_TILED(4,2);}else if(row_tile==4){LF_TILED(4,4);}else{LF_TILED(4,8);}
  }else{
   if(row_tile==2){LF_TILED(3,2);}else if(row_tile==4){LF_TILED(3,4);}else{LF_TILED(3,8);}
  }
#undef LF_TILED
 }else if(offs.numel()==5){
 expand_kernel<4><<<dim3(nblocks),dim3(threads),0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)z.data_ptr(),(uint16_t*)B.data_ptr(),boff.data_ptr<int64_t>(),bucket.data_ptr<int>(),(uint16_t*)y.data_ptr(),seg.data_ptr<int>(),wi.data_ptr<int>(),perm.data_ptr<int>(),ranks.data_ptr<int>(),scales.data_ptr<float>(),offs.data_ptr<int>(),nseg.data_ptr<int>(),(int)N,(int)(offs.numel()-1),(int)R,y.stride(0));
 }else{
 expand_kernel<3><<<dim3(nblocks),dim3(threads),0,at::cuda::getCurrentCUDAStream()>>>((uint16_t*)z.data_ptr(),(uint16_t*)B.data_ptr(),boff.data_ptr<int64_t>(),bucket.data_ptr<int>(),(uint16_t*)y.data_ptr(),seg.data_ptr<int>(),wi.data_ptr<int>(),perm.data_ptr<int>(),ranks.data_ptr<int>(),scales.data_ptr<float>(),offs.data_ptr<int>(),nseg.data_ptr<int>(),(int)N,(int)(offs.numel()-1),(int)R,y.stride(0));
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("shrink",&shrink);m.def("expand",&expand,
 pybind11::arg("z"),pybind11::arg("B"),pybind11::arg("boff"),pybind11::arg("bucket"),pybind11::arg("y"),pybind11::arg("seg"),pybind11::arg("wi"),pybind11::arg("perm"),pybind11::arg("ranks"),pybind11::arg("scales"),pybind11::arg("offs"),pybind11::arg("nseg"),pybind11::arg("N"),pybind11::arg("R"),pybind11::arg("nblocks"),pybind11::arg("threads"),pybind11::arg("row_tile")=16);}
