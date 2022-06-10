# 0610 learning summary

## Basic structure of the code

---

1. Initialization. Include some set-up and memory allocation.
   - This part can be done by CPU as calculation quantity of this part is much less than main loop.
   - All the `malloc` must be followed by `cudaMallocManaged` or be replaced by `cudaMalloc`, otherwise memory alloced can't be accessed by device.
2. Main loop to progress forward in time. (Two parts)
   1. (Part 1) Compute log of L2 norm of difference.
      - So far CPU handle this part.
   2. (Part 2) Evolve scalar wave initial data forward in time.
      - This is the most important part to transform the C code into cuda code.
3. Free the memory.
   - Use `cudaFree` instead of `Free.`

---

## Needed modifications

```c
//CUDA_basic_defines.h
#define LOOP_ALL_GFS_GPS_device(ii) for(int (ii)=gid;(ii)<Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2*NUM_EVOL_GFS;(ii)+=NUM_threads)
```

This defination can take the place of  the defination below in kernal and device code:

```c
//NRPy_basic_defines.h
#define LOOP_ALL_GFS_GPS(ii) _Pragma("omp parallel for") \
  for(int (ii)=0;(ii)<Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2*NUM_EVOL_GFS;(ii)++)
```

gid and NUM_threads' defination:

```c
//set_cudaparameters.h
const int NUM_threads = Grid_size * Block_size;

const int tx = threadIdx.x;
const int ty = threadIdx.y;
const int bx = blockIdx.x;
const int by = blockIdx.y;

const int blockid = by * Grid_size_x + bx;
const int tid = ty * Block_size_x + tx;
const int gid = blockid * Block_size + tid;
```

Code above are in `CUDA_basic_defines.h` and `set_cudaparameters.h`.

---

`rhs_eval_device` (device) function in `MoL_step_forward_in_time_GPU` (kernal)

```c
#include "./NRPy_basic_defines.h"
#include "./CUDA_basic_defines.h"


#define u_tile(tx, ty) u_tile[(ty)*SM_2D_Wx + (tx)]
#define tmp_tile(id, tx, ty) tmp_tile[(id)*SM_SIZE + (ty)*SM_2D_Wx + (tx)]
#define IN_GLOBAL_REGION(gm0, gm1, gm2) (IN_REGION((gm0), NGHOSTS, NGHOSTS+Nxx0) && IN_REGION((gm1), NGHOSTS, NGHOSTS+Nxx1) && IN_REGION((gm2), NGHOSTS, NGHOSTS+Nxx2))
#define IN_SM_REGION(tx, ty) (IN_REGION((tx), NGHOSTS, SM_2D_Wx - NGHOSTS) && IN_REGION((ty), NGHOSTS, SM_2D_Wy - NGHOSTS))

__device__ void rhs_eval_device(const paramstruct *restrict params, const rfm_struct *restrict rfmstruct, const REAL *restrict in_gfs, REAL *restrict rhs_gfs, REAL *restrict tmp_gm) {
  #include "./set_Cparameters.h"
  #include "./set_cudaparameters.h"
  const double FDPart1_Rational_2_3 = 2.0/3.0;
  const double FDPart1_Rational_1_12 = 1.0/12.0;
  const double FDPart1_Rational_5_2 = 5.0/2.0;
  const double FDPart1_Rational_4_3 = 4.0/3.0;
  __shared__ double u_tile[SM_SIZE];
  __shared__ double tmp_tile[SM_SIZE];
  

  //uu_dD0, uu_dD1, uu_dDD00, uu_dDD11
  {
    const int loopnum_0 = (int)( (Nxx0-1)/(Block_size_x - 2*NGHOSTS) + 1);
    const int loopnum_1 = (int)( (Nxx1-1)/(Block_size_y - 2*NGHOSTS) + 1);
    const int loopnum_2 = Nxx2;
    const int loopnum = loopnum_0 * loopnum_1 * loopnum_2;
    for(int b_iter = blockid; b_iter < loopnum; b_iter += Grid_size){
      const int biter_2 = (int) ( b_iter / (loopnum_0) / (loopnum_1));
      const int biter_1 = (int) ( (b_iter - biter_2 * (loopnum_0) * (loopnum_1)) / (loopnum_0));
      const int biter_0 = (int) ( (b_iter - biter_2 * (loopnum_0) * (loopnum_1)) % (loopnum_0));

      const int gm0 = biter_0 * (Block_size_x - 2*NGHOSTS) + tx;
      const int gm1 = biter_1 * (Block_size_y - 2*NGHOSTS) + ty;
      const int gm2 = NGHOSTS + biter_2;

      //load u to shared memory
      __syncthreads();
      if(IN_GLOBAL_REGION(gm0,gm1,gm2)){
        u_tile(tx, ty) = in_gfs[IDX4S(UUGF, gm0,gm1,gm2)];
      }
      else{
        u_tile(tx, ty) = 0;
      }__syncthreads();
      
      //some coefficients
      double FDPart1_1, FDPart1_2 ,uu;
      if(IN_SM_REGION(tx,ty)){
        uu = u_tile(tx, ty);
        FDPart1_1 = -u_tile(tx, ty+2);  
        FDPart1_2 = -FDPart1_Rational_5_2*uu;
      } __syncthreads();

      //uu_dD0
      {
        if(IN_SM_REGION(tx,ty)){
          tmp_tile(0, tx, ty) = invdx0*(FDPart1_Rational_1_12*(u_tile(tx-2,ty) - u_tile(tx+2,ty)) + FDPart1_Rational_2_3*(-u_tile(tx-1, ty) + u_tile(tx+1, ty)));
        } __syncthreads();
      
        if(IN_SM_REGION(tx,ty) && IN_GLOBAL_REGION(gm0,gm1,gm2)){
          tmp_gm[IDX4S(0, gm0, gm1, gm2)] = tmp_tile(0, tx, ty);
        }__syncthreads();
      }
      
      //uu_dD1
      {
        if(IN_SM_REGION(tx,ty)){
          tmp_tile[SM_2D_Wx * ty + tx] = invdx1*(FDPart1_Rational_1_12*(FDPart1_1 + u_tile(tx, ty-2)) + FDPart1_Rational_2_3*(-u_tile(tx, ty-1) + u_tile(tx, ty+1)));
        } __syncthreads();
      
        if(IN_SM_REGION(tx,ty) && IN_GLOBAL_REGION(gm0,gm1,gm2)){
          tmp_gm[IDX4S(1, gm0, gm1, gm2)] = tmp_tile(0, tx, ty);
        }__syncthreads();
      }

      //uu_dDD00
      {
        if(IN_SM_REGION(tx,ty)){
          tmp_tile[SM_2D_Wx * ty + tx] = ((invdx0)*(invdx0))*(FDPart1_2 + FDPart1_Rational_1_12*(-u_tile(tx-2, ty) - u_tile(tx+2, ty) ) + FDPart1_Rational_4_3*(+u_tile(tx-1, ty) + u_tile(tx+1, ty) ));
        } __syncthreads();
      
        if(IN_SM_REGION(tx,ty) && IN_GLOBAL_REGION(gm0,gm1,gm2)){
          tmp_gm[IDX4S(2, gm0, gm1, gm2)] = tmp_tile(0, tx, ty);
        }__syncthreads();
      }

      //uu_dDD11
      {
        if(IN_SM_REGION(tx,ty)){
          tmp_tile[SM_2D_Wx * ty + tx] = ((invdx1)*(invdx1))*(FDPart1_2 + FDPart1_Rational_1_12*(FDPart1_1 - u_tile(tx, ty-2)) + FDPart1_Rational_4_3*(u_tile(tx, ty-1) + u_tile(tx, ty+1)));
        } __syncthreads();
      
        if(IN_SM_REGION(tx,ty) && IN_GLOBAL_REGION(gm0,gm1,gm2)){
          tmp_gm[IDX4S(3, gm0, gm1, gm2)] = tmp_tile(0, tx, ty);
        }__syncthreads();
      }
    }
  }

  //uu_dDD22
  {
    const int loopnum_0 = (int)( (Nxx0-1)/(Block_size_x - 2*NGHOSTS) + 1);
    const int loopnum_1 = Nxx1; 
    const int loopnum_2 = (int)( (Nxx2-1)/(Block_size_y - 2*NGHOSTS) + 1);
    const int loopnum = loopnum_0 * loopnum_1 * loopnum_2;
    for(int b_iter = blockid; b_iter < loopnum; b_iter += Grid_size){
      const int biter_1 = (int) ( b_iter / (loopnum_0) / (loopnum_2));
      const int biter_2 = (int) ( (b_iter - biter_1 * (loopnum_0) * (loopnum_2)) / (loopnum_0));
      const int biter_0 = (int) ( (b_iter - biter_1 * (loopnum_0) * (loopnum_2)) % (loopnum_0));

      const int gm0 = biter_0 * (Block_size_x - 2*NGHOSTS) + tx;
      const int gm1 = NGHOSTS + biter_1;
      const int gm2 = biter_2 * (Block_size_y - 2*NGHOSTS) + ty; 

      //load u to shared memory
      __syncthreads();
      if(IN_GLOBAL_REGION(gm0,gm1,gm2)){
        u_tile(tx, ty) = in_gfs[IDX4S(UUGF, gm0,gm1,gm2)];
      }
      else{
        u_tile(tx, ty) = 0;
      }__syncthreads();
      
      //some coefficients
      double FDPart1_1, FDPart1_2, uu;
      if(IN_SM_REGION(tx,ty)){
        uu = u_tile(tx, ty);
        FDPart1_1 = -u_tile(tx, ty+2);  
        FDPart1_2 = -FDPart1_Rational_5_2*uu;
      } __syncthreads();

      //uu_dDD22
      {
        if(IN_SM_REGION(tx,ty)){
          tmp_tile[SM_2D_Wx * ty + tx] = ((invdx2)*(invdx2))*(FDPart1_2 + FDPart1_Rational_1_12*(-u_tile(tx, ty-2)- u_tile(tx, ty+2)) + FDPart1_Rational_4_3*(u_tile(tx, ty-1)+ u_tile(tx, ty+1)));
        } __syncthreads();
      
        if(IN_SM_REGION(tx,ty) && IN_GLOBAL_REGION(gm0,gm1,gm2)){
          tmp_gm[IDX4S(4, gm0, gm1, gm2)] = tmp_tile(0, tx, ty);
        }__syncthreads();
      }
    }
  }


  //total sumup
  {
    const int loopnum_0 = (int)( (Nxx0-1)/(Block_size_x) + 1);
    const int loopnum_1 = (int)( (Nxx1-1)/(Block_size_y) + 1);
    const int loopnum_2 = Nxx2;
    const int loopnum = loopnum_0 * loopnum_1 * loopnum_2;
    for(int b_iter = blockid; b_iter < loopnum; b_iter += Grid_size){
      const int biter_2 = (int) ( b_iter / (loopnum_0) / (loopnum_1));
      const int biter_1 = (int) ( (b_iter - biter_2 * (loopnum_0) * (loopnum_1)) / (loopnum_0));
      const int biter_0 = (int) ( (b_iter - biter_2 * (loopnum_0) * (loopnum_1)) % (loopnum_0));

      const int i0 = biter_0 * (Block_size_x) + tx;
      const int i1 = biter_1 * (Block_size_y) + ty;
      const int i2 = NGHOSTS + biter_2;


      double rhs_gfs_vv, vv;
      //load all derivatives in registers
      __syncthreads();
      if(IN_GLOBAL_REGION(i0,i1,i2)){
        const double uu = in_gfs[IDX4S(UUGF, i0,i1,i2)];
        vv = in_gfs[IDX4S(VVGF, i0,i1,i2)];
        const double uu_dD0 = tmp_gm[IDX4S(0, i0,i1,i2)];
        const double uu_dD1 = tmp_gm[IDX4S(1, i0,i1,i2)];
        const double uu_dDD00 = tmp_gm[IDX4S(2, i0,i1,i2)];
        const double uu_dDD11 = tmp_gm[IDX4S(3, i0,i1,i2)];
        const double uu_dDD22 = tmp_gm[IDX4S(4, i0,i1,i2)];
        #include "rfm_files/rfm_struct__read0.h"
        #include "rfm_files/rfm_struct__read1.h"
        #include "rfm_files/rfm_struct__read2.h"

        const double FDPart3_0 = (1.0/((f0_of_xx0)*(f0_of_xx0)));

        rhs_gfs_vv = ((wavespeed)*(wavespeed))*(FDPart3_0*uu_dDD11 + FDPart3_0*f1_of_xx1__D1*uu_dD1/f1_of_xx1 + FDPart3_0*uu_dDD22/((f1_of_xx1)*(f1_of_xx1)) - uu_dD0*(f0_of_xx0__DD00/((f0_of_xx0__D0)*(f0_of_xx0__D0)*(f0_of_xx0__D0)) - 2/(f0_of_xx0*f0_of_xx0__D0)) + uu_dDD00/((f0_of_xx0__D0)*(f0_of_xx0__D0)));
      }

      __syncthreads();
      if(IN_GLOBAL_REGION(i0,i1,i2)){
        rhs_gfs[IDX4S(UUGF, i0,i1,i2)] = vv;
        rhs_gfs[IDX4S(VVGF, i0,i1,i2)] = rhs_gfs_vv;
      }__syncthreads();
    }
  }
}

```

---

## Problems I can't solve

### Read-only problem

When I run the code, an error appeared at the fist loop:

``` 
CUDA error tag loop: an illegal memory access was encountered
```

And I use `cuda-memcheck` to find out how this error appears, and this the result using `cuda-memcheck`

```shell
========= CUDA-MEMCHECK
========= Invalid __global__ read of size 8
=========     at 0x00000028 in MoL_step_forward_in_time_GPU(__griddata__*, double, double*)
=========     by thread (7,3,0) in block (0,4,0)
=========     Address 0x7fffc72414f0 is out of bounds
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib/x86_64-linux-gnu/libcuda.so.1 [0x2082ca]
=========     Host Frame:./ScalarWaveCurvilinear_Playground [0x21a19]
=========     Host Frame:./ScalarWaveCurvilinear_Playground [0x21aa7]
=========     Host Frame:./ScalarWaveCurvilinear_Playground [0x57df5]
=========     Host Frame:./ScalarWaveCurvilinear_Playground [0x9885]
=========     Host Frame:./ScalarWaveCurvilinear_Playground [0x974c]
=========     Host Frame:./ScalarWaveCurvilinear_Playground [0x97a5]
=========     Host Frame:./ScalarWaveCurvilinear_Playground [0x149f6]
=========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (__libc_start_main + 0xf3) [0x24083]
=========     Host Frame:./ScalarWaveCurvilinear_Playground [0x7abe]
... every threads came out this error
```

Seem to be an out-of-bounds accessing, but after tests I found that GPU can read from the address but can't write data to the address. (I call it Read-only bug).

---

### Register not enough problem

When I use $32 \times 32$ blocks, cuda error:

```
CUDA error tag loop: too many resources request for lauch
```

This is because there are too many variables define in every threads, the number of registers are not enough to hold all the variables.

Simply reduce the block size can solve this problem.

 