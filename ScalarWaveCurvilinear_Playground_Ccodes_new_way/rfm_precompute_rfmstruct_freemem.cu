#include "././NRPy_basic_defines.h"
/*
 * Reference Metric Precomputation infrastructure: Free rfmstruct memory
 */
 __host__ void rfm_precompute_rfmstruct_freemem(const paramstruct *restrict params, rfm_struct *restrict rfmstruct) {
#include "./set_Cparameters.h"

  cudaFree(rfmstruct->f0_of_xx0);
  cudaFree(rfmstruct->f0_of_xx0__D0);
  cudaFree(rfmstruct->f0_of_xx0__DD00);
  cudaFree(rfmstruct->f0_of_xx0__DDD000);
  cudaFree(rfmstruct->f1_of_xx1);
  cudaFree(rfmstruct->f1_of_xx1__D1);
  cudaFree(rfmstruct->f1_of_xx1__DD11);
}
