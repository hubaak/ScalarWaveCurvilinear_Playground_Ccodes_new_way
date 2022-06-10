#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"
#include "./CUDA_basic_defines.h"

#include <stdio.h>
/*
 * // main() function:
 * // Step 0: Read command-line input, set up grid structure, allocate memory for gridfunctions, set up coordinates
 * // Step 1: Write test data to gridfunctions
 * // Step 2: Overwrite all data in ghost zones with NaNs
 * // Step 3: Apply curvilinear boundary conditions
 * // Step 4: Print gridfunction data after curvilinear boundary conditions have been applied
 * // Step 5: Free all allocated memory
 */
int main(int argc, const char *argv[]) {
  griddata_struct griddata;
  set_Cparameters_to_default(&griddata.params);

  // Step 0a: Read command-line input, error out if nonconformant
  if(argc != 4 || atoi(argv[1]) < NGHOSTS || atoi(argv[2]) < NGHOSTS || atoi(argv[3]) < NGHOSTS) {
    printf("Error: Expected one command-line argument: ./ScalarWaveCurvilinear_Playground Nx0 Nx1 Nx2,\n");
    printf("where Nx[0,1,2] is the number of grid points in the 0, 1, and 2 directions.\n");
    printf("Nx[] MUST BE larger than NGHOSTS (= %d)\n",NGHOSTS);
    exit(1);
  }
  // Step 0b: Set up numerical grid structure, first in space...
  const int Nxx[3] = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]) };
  if(Nxx[0]%2 != 0 || Nxx[1]%2 != 0 || Nxx[2]%2 != 0) {
    printf("Error: Cannot guarantee a proper cell-centered grid if number of grid cells not set to even number.\n");
    printf("       For example, in case of angular directions, proper symmetry zones will not exist.\n");
    exit(1);
  }
  // Step 0c: Set free parameters, overwriting Cparameters defaults
  //          by hand or with command-line input, as desired.
#include "free_parameters.h"

  // Step 0d: Uniform coordinate grids are stored to *xx[3]
  // Step 0d.i: Set bcstruct
  
  bc_struct bcstruct;
  {
    int EigenCoord = 1;
    // Step 0d.ii: Call set_Nxx_dxx_invdx_params__and__xx(), which sets
    //             params Nxx,Nxx_plus_2NGHOSTS,dxx,invdx, and xx[] for the
    //             chosen Eigen-CoordSystem.
    set_Nxx_dxx_invdx_params__and__xx(EigenCoord, Nxx, &griddata.params, griddata.xx);
    // Step 0e: Find ghostzone mappings; set up bcstruct
    driver_bcstruct(&griddata.params, &griddata.bcstruct, griddata.xx);
    // Step 0e.i: Free allocated space for xx[][] array
    for(int i=0;i<3;i++) cudaFree(griddata.xx[i]);
  }

  

  // Step 0f: Call set_Nxx_dxx_invdx_params__and__xx(), which sets
  //          params Nxx,Nxx_plus_2NGHOSTS,dxx,invdx, and xx[] for the
  //          chosen (non-Eigen) CoordSystem.
  int EigenCoord = 0;
  set_Nxx_dxx_invdx_params__and__xx(EigenCoord, Nxx, &griddata.params, griddata.xx);

  // Step 0g: Time coordinate parameters
  const REAL t_final =  0.7*domain_size; /* Final time is set so that at t=t_final,
                                          * data at the origin have not been corrupted
                                          * by the approximate outer boundary condition */

  // Step 0h: Set timestep based on smallest proper distance between gridpoints and CFL factor
  REAL dt = find_timestep(&griddata.params, griddata.xx, CFL_FACTOR);
  //printf("# Timestep set to = %e\n",(double)dt);
  int N_final = (int)(t_final / dt + 0.5); // The number of points in time.
                                           // Add 0.5 to account for C rounding down
                                           // typecasts to integers.
  int output_every_N = (int)((REAL)N_final/800.0);
  if(output_every_N == 0) output_every_N = 1;

  // Step 0i: Error out if the number of auxiliary gridfunctions outnumber evolved gridfunctions.
  //              This is a limitation of the RK method. You are always welcome to declare & allocate
  //              additional gridfunctions by hand.
  if(NUM_AUX_GFS > NUM_EVOL_GFS) {
    printf("Error: NUM_AUX_GFS > NUM_EVOL_GFS. Either reduce the number of auxiliary gridfunctions,\n");
    printf("       or allocate (malloc) by hand storage for *diagnostic_output_gfs. \n");
    exit(1);
  }
  catch_cuda_error("Initialization");

  // Step 0j: Declare struct for gridfunctions and allocate memory for y_n_gfs gridfunctions
  MoL_malloc_y_n_gfs(&griddata.params, &griddata.gridfuncs);

  // Step 0k: Set up precomputed reference metric arrays
  // Step 0k.i: Allocate space for precomputed reference metric arrays.
  rfm_struct rfmstruct;
  rfm_precompute_rfmstruct_malloc(&griddata.params, &griddata.rfmstruct);

  // Step 0k.ii: Define precomputed reference metric arrays.
  rfm_precompute_rfmstruct_define(&griddata.params, griddata.xx, &griddata.rfmstruct);

  

  // Step 0.l: Set up initial data to be exact solution at time=0:
  griddata.params.time = 0.0; exact_solution_all_points(&griddata.params, griddata.xx, griddata.gridfuncs.y_n_gfs);

  // Step 0.m: Allocate memory for non y_n_gfs. We do this here to free up
  //         memory for setting up initial data (for cases in which initial
  //         data setup is memory intensive.)
  MoL_malloc_non_y_n_gfs(&griddata.params, &griddata.gridfuncs);

  dim3 blocksize(Block_size_x, Block_size_y);
  dim3 gridsize(Grid_size_x, Grid_size_y);

  const int Nxx_plus_2NGHOSTS0 = griddata.params.Nxx_plus_2NGHOSTS0;  // grid::Nxx_plus_2NGHOSTS0
  const int Nxx_plus_2NGHOSTS1 = griddata.params.Nxx_plus_2NGHOSTS1;  // grid::Nxx_plus_2NGHOSTS1
  const int Nxx_plus_2NGHOSTS2 = griddata.params.Nxx_plus_2NGHOSTS2;  // grid::Nxx_plus_2NGHOSTS2
  const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2;
  REAL *restrict tmp_gm =(REAL *restrict) malloc(sizeof(REAL) * 6 * Nxx_plus_2NGHOSTS_tot);
  cudaMallocManaged((void **)&(tmp_gm), sizeof(REAL) * 6 * Nxx_plus_2NGHOSTS_tot, cudaMemAttachGlobal);

  catch_cuda_error("Allocate memory");

  printf("N_final = %d\n", N_final);
  for(int n=0;n<=N_final;n++)
    { // Main loop to progress forward in time.

    // Step 1: Set current time to correct value & compute exact solution
    griddata.params.time = ((REAL)n)*dt;

    // Step 2: Code validation: Compute log of L2 norm of difference
    //         between numerical and exact solutions:
    //   log_L2_Norm = log10( sqrt[Integral( [numerical - exact]^2 * dV)] ),
    //         where integral is within 30% of the grid outer boundary (domain_size)
    if(n%output_every_N == 0) {
      REAL integral = 0.0;
      REAL numpts   = 0.0;
      const int Nxx_plus_2NGHOSTS0 = griddata.params.Nxx_plus_2NGHOSTS0;
      const int Nxx_plus_2NGHOSTS1 = griddata.params.Nxx_plus_2NGHOSTS1;
      const int Nxx_plus_2NGHOSTS2 = griddata.params.Nxx_plus_2NGHOSTS2;
//#pragma omp parallel for reduction(+:integral,numpts)
      LOOP_REGION(NGHOSTS,Nxx_plus_2NGHOSTS0-NGHOSTS,
                  NGHOSTS,Nxx_plus_2NGHOSTS1-NGHOSTS,
                  NGHOSTS,Nxx_plus_2NGHOSTS2-NGHOSTS) {
        REAL xCart[3]; xx_to_Cart(&griddata.params,griddata.xx,i0,i1,i2, xCart);
        if(sqrt(xCart[0]*xCart[0] + xCart[1]*xCart[1] + xCart[2]*xCart[2]) < domain_size*0.3) {
          REAL uu_exact,vv_exact; exact_solution_single_point(&griddata.params,
                                                              griddata.xx[0][i0],griddata.xx[1][i1],griddata.xx[2][i2],
                                                              &uu_exact,&vv_exact);
          double num   = (double)griddata.gridfuncs.y_n_gfs[IDX4S(UUGF,i0,i1,i2)];
          double exact = (double)uu_exact;
          integral += (num - exact)*(num - exact);
          numpts   += 1.0;
        }
      }
      // Compute and output the log of the L2 norm.
      REAL log_L2_Norm = log10(1e-16 + sqrt(integral/numpts));  // 1e-16 + ... avoids log10(0)
      printf("%e %e\n",(double)griddata.params.time, log_L2_Norm);
    }
    
    // Step 3: Evolve scalar wave initial data forward in time using Method of Lines with RK4 algorithm,
    //         applying quadratic extrapolation outer boundary conditions.
    // Step 3.b: Step forward one timestep (t -> t+dt) in time using
    //           chosen RK-like MoL timestepping algorithm
    //MoL_step_forward_in_time(&griddata, dt);
    MoL_step_forward_in_time_GPU<<<gridsize, blocksize>>>(&griddata, dt, tmp_gm);
    cudaDeviceSynchronize();
    catch_cuda_error("In loop");
    
  } // End main loop to progress forward in time.

  // Step 4: Free all allocated memory
  rfm_precompute_rfmstruct_freemem(&griddata.params, &griddata.rfmstruct);

  freemem_bcstruct(&griddata.params, &griddata.bcstruct);
  MoL_free_memory_y_n_gfs(&griddata.params, &griddata.gridfuncs);
  MoL_free_memory_non_y_n_gfs(&griddata.params, &griddata.gridfuncs);
  for(int i=0;i<3;i++) cudaFree(griddata.xx[i]);
  return 0;
}
