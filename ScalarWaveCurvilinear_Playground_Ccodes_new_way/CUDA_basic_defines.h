
#ifndef CUDA_BASIC_DEFINES_H
#define CUDA_BASIC_DEFINES_H

#define SM_2D_Wx 8
#define SM_2D_Wy 8
#define SM_SIZE (SM_2D_Wx*SM_2D_Wy)

#define Block_size_x 8
#define Block_size_y 8
#define Block_size (Block_size_x * Block_size_y)

#define Grid_size_x 1
#define Grid_size_y 5
#define Grid_size (Grid_size_x * Grid_size_y)

#define LOOP_ALL_GFS_GPS_device(ii) for(int (ii)=gid;(ii)<Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2*NUM_EVOL_GFS;(ii)+=NUM_threads)

#endif