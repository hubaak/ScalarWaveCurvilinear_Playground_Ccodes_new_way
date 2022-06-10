#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"
/*
 * Exact solution at a single point. params.time==0 corresponds to the initial data.
 */
 __host__ void exact_solution_single_point(const paramstruct *restrict params,
                const REAL xx0, const REAL xx1, const REAL xx2,
                REAL *uu_exact, REAL *vv_exact) {
#include "./set_Cparameters.h"

  /*
   * NRPy+ Finite Difference Code Generation, Step 1 of 1: Evaluate SymPy expressions and write to main memory:
   */
  const double FDPart3_0 = (1.0/(SINHW));
  const double FDPart3_2 = AMPL*(exp(FDPart3_0*xx0) - exp(-FDPart3_0*xx0))/(exp(FDPart3_0) - exp(-FDPart3_0));
  const double FDPart3_3 = FDPart3_2*sin(xx1);
  const double FDPart3_4 = time*wavespeed - (FDPart3_2*kk2*cos(xx1) + FDPart3_3*kk0*cos(xx2) + FDPart3_3*kk1*sin(xx2))/sqrt(((kk0)*(kk0)) + ((kk1)*(kk1)) + ((kk2)*(kk2)));
  *uu_exact = 2 - sin(FDPart3_4);
  *vv_exact = -wavespeed*cos(FDPart3_4);
}
