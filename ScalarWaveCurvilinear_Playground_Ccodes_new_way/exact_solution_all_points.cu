#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"
/*
 * Exact solution at all points. params.time==0 corresponds to the initial data.
 */
 __host__ void exact_solution_all_points(const paramstruct *restrict params,REAL *restrict xx[3], REAL *restrict in_gfs) {
#include "./set_Cparameters.h"

  //#pragma omp parallel for
  for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
    const REAL xx2 = xx[2][i2];
    for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
      const REAL xx1 = xx[1][i1];
      for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++) {
        const REAL xx0 = xx[0][i0];
        exact_solution_single_point(params, xx0, xx1, xx2,
                   &in_gfs[IDX4S(UUGF,i0,i1,i2)], &in_gfs[IDX4S(VVGF,i0,i1,i2)]);
      } // END LOOP: for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++)
    } // END LOOP: for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++)
  } // END LOOP: for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++)
}
