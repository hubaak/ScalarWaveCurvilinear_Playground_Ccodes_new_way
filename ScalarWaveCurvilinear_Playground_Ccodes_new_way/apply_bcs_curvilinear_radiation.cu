#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"
/*
 * Compute 1st derivative finite-difference derivative with arbitrary upwind
 */
__device__ static inline REAL FD1_arbitrary_upwind_x0_dirn(const paramstruct *restrict params, const REAL *restrict gf,
    const int i0,const int i1,const int i2, const int offset) {
#include "./set_Cparameters.h"

  switch(offset) {
  case 0:
    return (+1.0/12.0*gf[IDX3S(i0-2,i1,i2)]
            -2.0/3.0*gf[IDX3S(i0-1,i1,i2)]
            +2.0/3.0*gf[IDX3S(i0+1,i1,i2)]
            -1.0/12.0*gf[IDX3S(i0+2,i1,i2)]) * invdx0;
  case 1:
    return (-1.0/4.0*gf[IDX3S(i0-1,i1,i2)]
            -5.0/6.0*gf[IDX3S(i0,i1,i2)]
            +3.0/2.0*gf[IDX3S(i0+1,i1,i2)]
            -1.0/2.0*gf[IDX3S(i0+2,i1,i2)]
            +1.0/12.0*gf[IDX3S(i0+3,i1,i2)]) * invdx0;
  case -1:
    return (-1.0/12.0*gf[IDX3S(i0-3,i1,i2)]
            +1.0/2.0*gf[IDX3S(i0-2,i1,i2)]
            -3.0/2.0*gf[IDX3S(i0-1,i1,i2)]
            +5.0/6.0*gf[IDX3S(i0,i1,i2)]
            +1.0/4.0*gf[IDX3S(i0+1,i1,i2)]) * invdx0;
  case 2:
    return (-25.0/12.0*gf[IDX3S(i0,i1,i2)]
            +4*gf[IDX3S(i0+1,i1,i2)]
            -3*gf[IDX3S(i0+2,i1,i2)]
            +4.0/3.0*gf[IDX3S(i0+3,i1,i2)]
            -1.0/4.0*gf[IDX3S(i0+4,i1,i2)]) * invdx0;
  case -2:
    return (+1.0/4.0*gf[IDX3S(i0-4,i1,i2)]
            -4.0/3.0*gf[IDX3S(i0-3,i1,i2)]
            +3*gf[IDX3S(i0-2,i1,i2)]
            -4*gf[IDX3S(i0-1,i1,i2)]
            +25.0/12.0*gf[IDX3S(i0,i1,i2)]) * invdx0;
  }
  return 0.0 / 0.0;  // poison output if offset computed incorrectly
}
/*
 * Compute r(xx0,xx1,xx2).
 */
__device__  static inline void r_and_partial_xi_r_derivs(const paramstruct *restrict params,const REAL xx0,const REAL xx1,const REAL xx2,
                                  REAL *r, REAL *rinv, REAL *partial_x0_r,REAL *partial_x1_r,REAL *partial_x2_r) {
#include "./set_Cparameters.h"

  const double tmp_0 = (1.0/(SINHW));
  const double tmp_1 = exp(tmp_0) - exp(-tmp_0);
  const double tmp_3 = exp(tmp_0*xx0);
  const double tmp_4 = exp(-tmp_0*xx0);
  const double tmp_5 = tmp_3 - tmp_4;
  const double tmp_6 = tmp_1/AMPL;
  *r = AMPL*tmp_5/tmp_1;
  *rinv = tmp_6/tmp_5;
  *partial_x0_r = tmp_6/(tmp_0*tmp_3 + tmp_0*tmp_4);
  *partial_x1_r = 0;
  *partial_x2_r = 0;
}
/*
 * Compute \partial_r f
 */
__device__ static inline REAL compute_partial_r_f(const paramstruct *restrict params, REAL *restrict xx[3], const REAL *restrict gfs,
                                       const int which_gf, const int dest_i0,const int dest_i1,const int dest_i2,
                                       const int FACEi0,const int FACEi1,const int FACEi2,
                                       const REAL partial_x0_r, const REAL partial_x1_r, const REAL partial_x2_r) {
#include "./set_Cparameters.h"

  ///////////////////////////////////////////////////////////

  // FD1_stencil_radius = BC_FDORDER/2 = 2
  const int FD1_stencil_radius = 2;

  const int ntot = Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2;

  ///////////////////////////////////////////////////////////
  // Next we'll compute partial_xi f, using a maximally-centered stencil.
  //   The {i0,i1,i2}_offset parameters set the offset of the maximally-centered
  //   stencil, such that an offset=0 implies a centered stencil.

  // CHECK: Nxx_plus_2NGHOSTS0=10; FD1_stencil_radius=2. Then Nxx_plus_2NGHOSTS0-FD1_stencil_radius-1 = 7
  //  if dest_i0 = 9, we get i0_offset=7-9=-2, so the (4th order) deriv
  //  stencil is: -4,-3,-2,-1,0

  // CHECK: if FD1_stencil_radius=2 and dest_i0 = 1, we get i0_offset = FD1_stencil_radius-dest_i0 = 1,
  //  so the (4th order) deriv stencil is: -1,0,1,2,3

  // CHECK: if FD1_stencil_radius=2 and dest_i0 = 0, we get i0_offset = FD1_stencil_radius-1 = 2,
  //  so the (4th order) deriv stencil is: 0,1,2,3,4
  int i0_offset = FACEi0;  // up/downwind on the faces. This offset should never go out of bounds.
  if(dest_i0 < FD1_stencil_radius) i0_offset = FD1_stencil_radius-dest_i0;
  else if(dest_i0 > (Nxx_plus_2NGHOSTS0-FD1_stencil_radius-1)) i0_offset = (Nxx_plus_2NGHOSTS0-FD1_stencil_radius-1) - dest_i0;
  const REAL partial_x0_f=FD1_arbitrary_upwind_x0_dirn(params,&gfs[which_gf*ntot],dest_i0,dest_i1,dest_i2,i0_offset);

  const REAL partial_x1_f=0.0;
  const REAL partial_x2_f=0.0;
  return partial_x0_r*partial_x0_f + partial_x1_r*partial_x1_f + partial_x2_r*partial_x2_f;
}

/*
 * *** Apply radiation BCs to all outer boundaries. ***
 */
__device__ static inline void radiation_bcs_curvilinear(const paramstruct *restrict params, const bc_struct *restrict bcstruct,REAL *restrict xx[3],
                           const REAL *restrict gfs, REAL *restrict gfs_rhss,
                           const int which_gf, const int dest_i0,const int dest_i1,const int dest_i2,
                           const int FACEi0,const int FACEi1,const int FACEi2) {
#include "./set_Cparameters.h"
static const REAL gridfunctions_f_infinity_device[NUM_EVOL_GFS] = { 0.0, 0.0 };
static const REAL gridfunctions_wavespeed_device[NUM_EVOL_GFS] = { 1.0, 1.0 };


  // Nearest "interior" neighbor of this gridpoint, based on current face
  const int dest_i0_int=dest_i0+1*FACEi0, dest_i1_int=dest_i1+1*FACEi1, dest_i2_int=dest_i2+1*FACEi2;
  REAL r,rinv, partial_x0_r,partial_x1_r,partial_x2_r;
  REAL r_int,r_intinv, partial_x0_r_int,partial_x1_r_int,partial_x2_r_int;
  r_and_partial_xi_r_derivs(params,xx[0][dest_i0],    xx[1][dest_i1],    xx[2][dest_i2],    &r,    &rinv,    &partial_x0_r,    &partial_x1_r,    &partial_x2_r);
  r_and_partial_xi_r_derivs(params,xx[0][dest_i0_int],xx[1][dest_i1_int],xx[2][dest_i2_int],&r_int,&r_intinv,&partial_x0_r_int,&partial_x1_r_int,&partial_x2_r_int);
  const REAL partial_r_f     = compute_partial_r_f(params,xx,gfs, which_gf,dest_i0,    dest_i1,    dest_i2,
                                                   FACEi0,FACEi1,FACEi2,
                                                   partial_x0_r,    partial_x1_r,    partial_x2_r);
  const REAL partial_r_f_int = compute_partial_r_f(params,xx,gfs, which_gf,dest_i0_int,dest_i1_int,dest_i2_int,
                                                   FACEi0,FACEi1,FACEi2,
                                                   partial_x0_r_int,partial_x1_r_int,partial_x2_r_int);

  const int idx3 = IDX3S(dest_i0,dest_i1,dest_i2);
  const int idx3_int = IDX3S(dest_i0_int,dest_i1_int,dest_i2_int);

  const REAL partial_t_f_int = gfs_rhss[IDX4ptS(which_gf, idx3_int)];

  const REAL c = gridfunctions_wavespeed_device[which_gf];
  const REAL f_infinity = gridfunctions_f_infinity_device[which_gf];
  const REAL f = gfs[IDX4ptS(which_gf, idx3)];
  const REAL f_int = gfs[IDX4ptS(which_gf, idx3_int)];
  const REAL partial_t_f_int_outgoing_wave = -c * (partial_r_f_int + (f_int - f_infinity) * r_intinv);

  const REAL k = r_int*r_int*r_int * (partial_t_f_int - partial_t_f_int_outgoing_wave);

  const REAL partial_t_f_outgoing_wave = -c * (partial_r_f + (f - f_infinity) * rinv);

  gfs_rhss[IDX4ptS(which_gf, idx3)] = partial_t_f_outgoing_wave + k * rinv*rinv*rinv;
}

/*
 * Curvilinear boundary condition driver routine: Apply BCs to all six
 *   boundary faces of the 3D numerical domain, filling in the
 *   innermost ghost zone layer first, and moving outward.
 */
__device__  void apply_bcs_curvilinear_radiation(const paramstruct *restrict params, const bc_struct *restrict bcstruct,
                           const int NUM_GFS, const int8_t *restrict gfs_parity, REAL *restrict xx[3],
                           REAL *restrict gfs, REAL *restrict gfs_rhss) {
#include "./set_Cparameters.h"
#include "./set_cudaparameters.h"
  for(int which_gf=blockid; which_gf<NUM_GFS;which_gf+= Grid_size) {
    for(int which_gz = 0; which_gz < NGHOSTS; which_gz++) {

      // First apply OUTER boundary conditions,
      //   in case an INNER (parity) boundary point
      //   needs data at the outer boundary:
      // After updating each face, adjust imin[] and imax[]
      //   to reflect the newly-updated face extents.
      for(int pt=tid;pt<bcstruct->num_ob_gz_pts[which_gz];pt+=Block_size) {
        // *** Apply radiation BCs to all outer boundary points. ***
        radiation_bcs_curvilinear(params, bcstruct, xx, gfs, gfs_rhss,  which_gf,
                                  bcstruct->outer[which_gz][pt].outer_bc_dest_pt.i0,
                                  bcstruct->outer[which_gz][pt].outer_bc_dest_pt.i1,
                                  bcstruct->outer[which_gz][pt].outer_bc_dest_pt.i2,
                                  bcstruct->outer[which_gz][pt].FACEi0,
                                  bcstruct->outer[which_gz][pt].FACEi1,
                                  bcstruct->outer[which_gz][pt].FACEi2);
      }

      // Apply INNER (parity) boundary conditions:
      for(int pt=tid;pt<bcstruct->num_ib_gz_pts[which_gz];pt+=Block_size) {
        const int i0dest = bcstruct->inner[which_gz][pt].inner_bc_dest_pt.i0;
        const int i1dest = bcstruct->inner[which_gz][pt].inner_bc_dest_pt.i1;
        const int i2dest = bcstruct->inner[which_gz][pt].inner_bc_dest_pt.i2;
        const int i0src  = bcstruct->inner[which_gz][pt].inner_bc_src_pt.i0;
        const int i1src  = bcstruct->inner[which_gz][pt].inner_bc_src_pt.i1;
        const int i2src  = bcstruct->inner[which_gz][pt].inner_bc_src_pt.i2;
        gfs_rhss[IDX4S(which_gf,i0dest,i1dest,i2dest)] =
          bcstruct->inner[which_gz][pt].parity[gfs_parity[which_gf]] * gfs_rhss[IDX4S(which_gf, i0src,i1src,i2src)];
      } // END for(int pt=0;pt<num_ib_gz_pts[which_gz];pt++)
    } // END for(int which_gz = 0; which_gz < NGHOSTS; which_gz++)
  } // END for(int which_gf=0;which_gf<NUM_GFS;which_gf++)
}
