#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"
/*
 * Free memory allocated within bcstruct
 */
 __host__ void freemem_bcstruct(const paramstruct *restrict params, const bc_struct *restrict bcstruct) {
#include "./set_Cparameters.h"

  for(int i=0;i<NGHOSTS;i++) { cudaFree(bcstruct->outer[i]);  cudaFree(bcstruct->inner[i]); }
  cudaFree(bcstruct->outer);  cudaFree(bcstruct->inner);
  cudaFree(bcstruct->num_ob_gz_pts); cudaFree(bcstruct->num_ib_gz_pts);
}
