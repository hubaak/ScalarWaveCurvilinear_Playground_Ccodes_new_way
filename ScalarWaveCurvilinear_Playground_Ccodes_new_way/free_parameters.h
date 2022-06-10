
// Free parameters related to physical system:
griddata.params.time = 0.0; // Initial simulation time corresponds to exact solution at time=0.
griddata.params.wavespeed = 1.0;

// Free parameters related to numerical timestep:
REAL CFL_FACTOR = 1.0;

// Set free-parameter values.

const REAL domain_size    = 10.0;
const REAL sinh_width     = 0.4;
const REAL sinhv2_const_dr= 0.05;
const REAL SymTP_bScale   = 1.0;

griddata.params.AMPL = domain_size;
griddata.params.SINHW=  sinh_width;

