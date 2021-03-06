#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"
/*
 * Method of Lines (MoL) for "RK4" method: Free memory for "y_n_gfs" gridfunctions
 *    - y_n_gfs are used to store data for the vector of gridfunctions y_i at t_n, at the start of each MoL timestep
 *    - non_y_n_gfs are needed for intermediate (e.g., k_i) storage in chosen MoL method
 */
 __host__ void MoL_free_memory_y_n_gfs(const paramstruct *restrict params, MoL_gridfunctions_struct *restrict gridfuncs) {
#include "./set_Cparameters.h"

      cudaFree(gridfuncs->y_n_gfs);
}
