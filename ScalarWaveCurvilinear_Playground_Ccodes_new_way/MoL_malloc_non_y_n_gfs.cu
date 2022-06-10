#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"
/*
 * Method of Lines (MoL) for "RK4" method: Allocate memory for "non_y_n_gfs" gridfunctions
 *    * y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
 *    * non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method
 */
__host__ void MoL_malloc_non_y_n_gfs(const paramstruct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs) {
#include "./set_Cparameters.h"
  
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2;
  /*
  gridfuncs->y_nplus1_running_total_gfs = (REAL *restrict)malloc(sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  gridfuncs->k_odd_gfs = (REAL *restrict)malloc(sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  gridfuncs->k_even_gfs = (REAL *restrict)malloc(sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  gridfuncs->auxevol_gfs = (REAL *restrict)malloc(sizeof(REAL) * NUM_AUXEVOL_GFS * Nxx_plus_2NGHOSTS_tot);

  cudaMallocManaged((void **)&(gridfuncs->y_nplus1_running_total_gfs), sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot, cudaMemAttachGlobal);
  cudaMallocManaged((void **)&(gridfuncs->k_odd_gfs), sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot, cudaMemAttachGlobal);
  cudaMallocManaged((void **)&(gridfuncs->k_even_gfs), sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot, cudaMemAttachGlobal);  
  cudaMallocManaged((void **)&(gridfuncs->auxevol_gfs), sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot, cudaMemAttachGlobal);  
  */
  printf("y_nplus1_running_total_gfs = %lld\n", (long long int) gridfuncs->y_nplus1_running_total_gfs);
  printf("k_odd_gfs = %lld\n", (long long int) gridfuncs->k_odd_gfs);
  printf("k_even_gfs = %lld\n", (long long int) gridfuncs->k_even_gfs);
  cudaMalloc((void **)&(gridfuncs->y_nplus1_running_total_gfs), sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaMalloc((void **)&(gridfuncs->k_odd_gfs), sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaMalloc((void **)&(gridfuncs->k_even_gfs), sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaMalloc((void **)&(gridfuncs->auxevol_gfs), sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);

  printf("y_nplus1_running_total_gfs = %lld\n", (long long int) gridfuncs->y_nplus1_running_total_gfs);
  printf("k_odd_gfs = %lld\n", (long long int) gridfuncs->k_odd_gfs);
  printf("k_even_gfs = %lld\n", (long long int) gridfuncs->k_even_gfs);
  
  
  cudaMemset(gridfuncs->y_nplus1_running_total_gfs, 0, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaMemset(gridfuncs->k_odd_gfs, 0, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaMemset(gridfuncs->k_even_gfs, 0, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  cudaMemset(gridfuncs->auxevol_gfs, 0, sizeof(REAL) * NUM_EVOL_GFS * Nxx_plus_2NGHOSTS_tot);
  

  gridfuncs->diagnostic_output_gfs = gridfuncs->y_nplus1_running_total_gfs;
}
