#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <cooperative_groups.h>
/*
 * Method of Lines (MoL) for "RK4" method: Step forward one full timestep.
 */

__global__ void MoL_step_forward_in_time_GPU(griddata_struct *griddata, const REAL dt, REAL *tmp_gm) {
  #include "./set_cudaparameters.h"
  // C code implementation of -={ RK4 }=- Method of Lines timestepping.
  static const int8_t evol_gf_parity_device[2] = { 0, 0 };
  // -={ START k1 substep }=-
  {
    // Set gridfunction aliases from gridfuncs struct
    REAL *restrict y_n_gfs = griddata->gridfuncs.y_n_gfs;  // y_n gridfunctions
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = griddata->gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = griddata->gridfuncs.k_odd_gfs;
    REAL *restrict k_even_gfs = griddata->gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata->gridfuncs.auxevol_gfs;
    paramstruct *restrict params = &griddata->params;
    const rfm_struct *restrict rfmstruct = &griddata->rfmstruct;
    const bc_struct *restrict bcstruct = &griddata->bcstruct;
    const int Nxx_plus_2NGHOSTS0 = griddata->params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata->params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata->params.Nxx_plus_2NGHOSTS2;
    rhs_eval_device(params, rfmstruct, y_n_gfs, k_odd_gfs, tmp_gm);
    //apply_bcs_curvilinear_radiation(params, bcstruct, NUM_EVOL_GFS, evol_gf_parity_device, griddata->xx, y_n_gfs, k_odd_gfs);
    
    __syncthreads();

    LOOP_ALL_GFS_GPS_device(i) {
      const REAL k_odd_gfsL = k_odd_gfs[i];
      const REAL y_n_gfsL = y_n_gfs[i];
      //k_odd_gfs[i] = 0;
      //to do: try to prefetch them
      //y_nplus1_running_total_gfs[i] = (1.0/6.0)*dt*k_odd_gfsL;
      //k_odd_gfs[i] = (1.0/2.0)*dt*k_odd_gfsL + y_n_gfsL;
    }
  }
  // -={ END k1 substep }=-

  // -={ START k2 substep }=-
  {
    // Set gridfunction aliases from gridfuncs struct
    REAL *restrict y_n_gfs = griddata->gridfuncs.y_n_gfs;  // y_n gridfunctions
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = griddata->gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = griddata->gridfuncs.k_odd_gfs;
    REAL *restrict k_even_gfs = griddata->gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata->gridfuncs.auxevol_gfs;
    paramstruct *restrict params = &griddata->params;
    const rfm_struct *restrict rfmstruct = &griddata->rfmstruct;
    const bc_struct *restrict bcstruct = &griddata->bcstruct;
    const int Nxx_plus_2NGHOSTS0 = griddata->params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata->params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata->params.Nxx_plus_2NGHOSTS2;
    //rhs_eval_device(params, rfmstruct, k_odd_gfs, k_even_gfs, tmp_gm);apply_bcs_curvilinear_radiation(params, bcstruct, NUM_EVOL_GFS, evol_gf_parity_device, griddata->xx, k_odd_gfs, k_even_gfs);
    /*
    LOOP_ALL_GFS_GPS_device(i) {
      const REAL k_even_gfsL = k_even_gfs[i];
      const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
      const REAL y_n_gfsL = y_n_gfs[i];
      y_nplus1_running_total_gfs[i] = (1.0/3.0)*dt*k_even_gfsL + y_nplus1_running_total_gfsL;
      k_even_gfs[i] = (1.0/2.0)*dt*k_even_gfsL + y_n_gfsL;
    }*/
  }
  // -={ END k2 substep }=-

  // -={ START k3 substep }=-
  {
    // Set gridfunction aliases from gridfuncs struct
    REAL *restrict y_n_gfs = griddata->gridfuncs.y_n_gfs;  // y_n gridfunctions
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = griddata->gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = griddata->gridfuncs.k_odd_gfs;
    REAL *restrict k_even_gfs = griddata->gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata->gridfuncs.auxevol_gfs;
    paramstruct *restrict params = &griddata->params;
    const rfm_struct *restrict rfmstruct = &griddata->rfmstruct;
    const bc_struct *restrict bcstruct = &griddata->bcstruct;
    const int Nxx_plus_2NGHOSTS0 = griddata->params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata->params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata->params.Nxx_plus_2NGHOSTS2;
    //rhs_eval_device(params, rfmstruct, k_even_gfs, k_odd_gfs, tmp_gm);apply_bcs_curvilinear_radiation(params, bcstruct, NUM_EVOL_GFS, evol_gf_parity_device, griddata->xx, k_even_gfs, k_odd_gfs);
    /*
    LOOP_ALL_GFS_GPS_device(i) {
      const REAL k_odd_gfsL = k_odd_gfs[i];
      const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
      const REAL y_n_gfsL = y_n_gfs[i];
      y_nplus1_running_total_gfs[i] = (1.0/3.0)*dt*k_odd_gfsL + y_nplus1_running_total_gfsL;
      k_odd_gfs[i] = dt*k_odd_gfsL + y_n_gfsL;
    }*/
  }
  // -={ END k3 substep }=-

  // -={ START k4 substep }=-
  {
    // Set gridfunction aliases from gridfuncs struct
    REAL *restrict y_n_gfs = griddata->gridfuncs.y_n_gfs;  // y_n gridfunctions
    // Temporary timelevel & AUXEVOL gridfunctions:
    REAL *restrict y_nplus1_running_total_gfs = griddata->gridfuncs.y_nplus1_running_total_gfs;
    REAL *restrict k_odd_gfs = griddata->gridfuncs.k_odd_gfs;
    REAL *restrict k_even_gfs = griddata->gridfuncs.k_even_gfs;
    REAL *restrict auxevol_gfs = griddata->gridfuncs.auxevol_gfs;
    paramstruct *restrict params = &griddata->params;
    const rfm_struct *restrict rfmstruct = &griddata->rfmstruct;
    const bc_struct *restrict bcstruct = &griddata->bcstruct;
    const int Nxx_plus_2NGHOSTS0 = griddata->params.Nxx_plus_2NGHOSTS0;
    const int Nxx_plus_2NGHOSTS1 = griddata->params.Nxx_plus_2NGHOSTS1;
    const int Nxx_plus_2NGHOSTS2 = griddata->params.Nxx_plus_2NGHOSTS2;
    //rhs_eval_device(params, rfmstruct, k_odd_gfs, k_even_gfs, tmp_gm);apply_bcs_curvilinear_radiation(params, bcstruct, NUM_EVOL_GFS, evol_gf_parity_device, griddata->xx, k_odd_gfs, k_even_gfs);
    /*
    LOOP_ALL_GFS_GPS_device(i) {
      const REAL k_even_gfsL = k_even_gfs[i];
      const REAL y_n_gfsL = y_n_gfs[i];
      const REAL y_nplus1_running_total_gfsL = y_nplus1_running_total_gfs[i];
      y_n_gfs[i] = (1.0/6.0)*dt*k_even_gfsL + y_n_gfsL + y_nplus1_running_total_gfsL;
    }*/
  }
  // -={ END k4 substep }=-

}