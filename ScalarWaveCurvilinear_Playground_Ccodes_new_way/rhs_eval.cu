#include "./NRPy_basic_defines.h"
#include "./CUDA_basic_defines.h"


#define u_tile(tx, ty) u_tile[(ty)*SM_2D_Wx + (tx)]
#define tmp_tile(id, tx, ty) tmp_tile[(id)*SM_SIZE + (ty)*SM_2D_Wx + (tx)]
#define IN_GLOBAL_REGION(gm0, gm1, gm2) (IN_REGION((gm0), NGHOSTS, NGHOSTS+Nxx0) && IN_REGION((gm1), NGHOSTS, NGHOSTS+Nxx1) && IN_REGION((gm2), NGHOSTS, NGHOSTS+Nxx2))
#define IN_SM_REGION(tx, ty) (IN_REGION((tx), NGHOSTS, SM_2D_Wx - NGHOSTS) && IN_REGION((ty), NGHOSTS, SM_2D_Wy - NGHOSTS))
/*
 * Evaluate the scalar wave RHSs
 */
__host__ void rhs_eval(const paramstruct *restrict params, const rfm_struct *restrict rfmstruct, const REAL *restrict in_gfs, REAL *restrict rhs_gfs) {
  #include "./set_Cparameters.h"

  #pragma omp parallel for
  //why not use LOOP_REGION here?
  //there #include in the loop
  for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++) {
    #include "rfm_files/rfm_struct__read2.h"
    for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++) {
      #include "rfm_files/rfm_struct__read1.h"
      for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++) {
        #include "rfm_files/rfm_struct__read0.h"
        {
          /*
           * NRPy+ Finite Difference Code Generation, Step 1 of 2: Read from main memory and compute finite difference stencils:
           */
          /*
           *  Original SymPy expressions:
           *  "[const double uu_dD0 = invdx0*(-2*uu_i0m1_i1_i2/3 + uu_i0m2_i1_i2/12 + 2*uu_i0p1_i1_i2/3 - uu_i0p2_i1_i2/12),
           *    const double uu_dD1 = invdx1*(-2*uu_i0_i1m1_i2/3 + uu_i0_i1m2_i2/12 + 2*uu_i0_i1p1_i2/3 - uu_i0_i1p2_i2/12),
           *    const double uu_dDD00 = invdx0**2*(-5*uu/2 + 4*uu_i0m1_i1_i2/3 - uu_i0m2_i1_i2/12 + 4*uu_i0p1_i1_i2/3 - uu_i0p2_i1_i2/12),
           *    const double uu_dDD11 = invdx1**2*(-5*uu/2 + 4*uu_i0_i1m1_i2/3 - uu_i0_i1m2_i2/12 + 4*uu_i0_i1p1_i2/3 - uu_i0_i1p2_i2/12),
           *    const double uu_dDD22 = invdx2**2*(-5*uu/2 + 4*uu_i0_i1_i2m1/3 - uu_i0_i1_i2m2/12 + 4*uu_i0_i1_i2p1/3 - uu_i0_i1_i2p2/12)]"
           */
           
          const double uu_i0_i1_i2m2 = in_gfs[IDX4S(UUGF, i0,i1,i2-2)];
          const double uu_i0_i1_i2m1 = in_gfs[IDX4S(UUGF, i0,i1,i2-1)];
          const double uu_i0_i1m2_i2 = in_gfs[IDX4S(UUGF, i0,i1-2,i2)];
          const double uu_i0_i1m1_i2 = in_gfs[IDX4S(UUGF, i0,i1-1,i2)];
          const double uu_i0m2_i1_i2 = in_gfs[IDX4S(UUGF, i0-2,i1,i2)];
          const double uu_i0m1_i1_i2 = in_gfs[IDX4S(UUGF, i0-1,i1,i2)];
          const double uu = in_gfs[IDX4S(UUGF, i0,i1,i2)];
          const double uu_i0p1_i1_i2 = in_gfs[IDX4S(UUGF, i0+1,i1,i2)];
          const double uu_i0p2_i1_i2 = in_gfs[IDX4S(UUGF, i0+2,i1,i2)];
          const double uu_i0_i1p1_i2 = in_gfs[IDX4S(UUGF, i0,i1+1,i2)];
          const double uu_i0_i1p2_i2 = in_gfs[IDX4S(UUGF, i0,i1+2,i2)];
          const double uu_i0_i1_i2p1 = in_gfs[IDX4S(UUGF, i0,i1,i2+1)];
          const double uu_i0_i1_i2p2 = in_gfs[IDX4S(UUGF, i0,i1,i2+2)];
          const double vv = in_gfs[IDX4S(VVGF, i0,i1,i2)];

          const double FDPart1_Rational_2_3 = 2.0/3.0;
          const double FDPart1_Rational_1_12 = 1.0/12.0;
          const double FDPart1_Rational_5_2 = 5.0/2.0;
          const double FDPart1_Rational_4_3 = 4.0/3.0;
          const double FDPart1_1 = -uu_i0_i1p2_i2;
          const double FDPart1_2 = -FDPart1_Rational_5_2*uu;
          const double uu_dD0 = invdx0*(FDPart1_Rational_1_12*(uu_i0m2_i1_i2 - uu_i0p2_i1_i2) + FDPart1_Rational_2_3*(-uu_i0m1_i1_i2 + uu_i0p1_i1_i2));
          const double uu_dD1 = invdx1*(FDPart1_Rational_1_12*(FDPart1_1 + uu_i0_i1m2_i2) + FDPart1_Rational_2_3*(-uu_i0_i1m1_i2 + uu_i0_i1p1_i2));
          const double uu_dDD00 = ((invdx0)*(invdx0))*(FDPart1_2 + FDPart1_Rational_1_12*(-uu_i0m2_i1_i2 - uu_i0p2_i1_i2) + FDPart1_Rational_4_3*(uu_i0m1_i1_i2 + uu_i0p1_i1_i2));

          const double uu_dDD11 = ((invdx1)*(invdx1))*(FDPart1_2 + FDPart1_Rational_1_12*(FDPart1_1 - uu_i0_i1m2_i2) + FDPart1_Rational_4_3*(uu_i0_i1m1_i2 + uu_i0_i1p1_i2));
          const double uu_dDD22 = ((invdx2)*(invdx2))*(FDPart1_2 + FDPart1_Rational_1_12*(-uu_i0_i1_i2m2 - uu_i0_i1_i2p2) + FDPart1_Rational_4_3*(uu_i0_i1_i2m1 + uu_i0_i1_i2p1));
          /*
           * NRPy+ Finite Difference Code Generation, Step 2 of 2: Evaluate SymPy expressions and write to main memory:
           */
          /*
           *  Original SymPy expressions:
           *  "[rhs_gfs[IDX4S(UUGF, i0, i1, i2)] = vv,
           *    rhs_gfs[IDX4S(VVGF, i0, i1, i2)] = wavespeed**2*(-uu_dD0*(f0_of_xx0__DD00/f0_of_xx0__D0**3 - 2/(f0_of_xx0*f0_of_xx0__D0)) + uu_dDD00/f0_of_xx0__D0**2 + uu_dDD11/f0_of_xx0**2 + f1_of_xx1__D1*uu_dD1/(f0_of_xx0**2*f1_of_xx1) + uu_dDD22/(f0_of_xx0**2*f1_of_xx1**2))]"
           */
          const double FDPart3_0 = (1.0/((f0_of_xx0)*(f0_of_xx0)));
          rhs_gfs[IDX4S(UUGF, i0, i1, i2)] = vv;
          rhs_gfs[IDX4S(VVGF, i0, i1, i2)] = ((wavespeed)*(wavespeed))*(FDPart3_0*uu_dDD11 + FDPart3_0*f1_of_xx1__D1*uu_dD1/f1_of_xx1 + FDPart3_0*uu_dDD22/((f1_of_xx1)*(f1_of_xx1)) - uu_dD0*(f0_of_xx0__DD00/((f0_of_xx0__D0)*(f0_of_xx0__D0)*(f0_of_xx0__D0)) - 2/(f0_of_xx0*f0_of_xx0__D0)) + uu_dDD00/((f0_of_xx0__D0)*(f0_of_xx0__D0)));
        }
      } // END LOOP: for (int i0 = NGHOSTS; i0 < NGHOSTS+Nxx0; i0++)
    } // END LOOP: for (int i1 = NGHOSTS; i1 < NGHOSTS+Nxx1; i1++)
  } // END LOOP: for (int i2 = NGHOSTS; i2 < NGHOSTS+Nxx2; i2++)
}

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