CC     = nvcc
CFLAGS = -Xcompiler -fopenmp -rdc=true -Xptxas -v
#CFLAGS =   -O2 -march=native -fopenmp -Wall -Wno-unused-variable -std=gnu99
#CFLAGS =  -O2 -march=native -g -fopenmp -Wall -Wno-unused-variable -std=gnu99
#CFLAGS =  -O2 -g -Wall -Wno-unused-variable -Wno-unknown-pragmas -std=gnu99
#CFLAGS =   -O2 -march=native -fopenmp -Wall -Wno-unused-variable -std=gnu99
INCLUDEDIRS = 
all: ScalarWaveCurvilinear_Playground  MoL_malloc_y_n_gfs.o MoL_free_memory_y_n_gfs.o MoL_malloc_non_y_n_gfs.o MoL_free_memory_non_y_n_gfs.o MoL_step_forward_in_time.o set_bcstruct.o driver_bcstruct.o apply_bcs_curvilinear_inner_only.o freemem_bcstruct.o set_up__bc_gz_map_and_parity_condns.o apply_bcs_curvilinear_radiation.o find_timestep.o xx_to_Cart.o set_Nxx_dxx_invdx_params__and__xx.o Cart_to_xx_and_nearest_i0i1i2.o Cart_to_xx_and_nearest_i0i1i2_global_grid_center.o rfm_precompute_rfmstruct_malloc.o rfm_precompute_rfmstruct_define.o rfm_precompute_rfmstruct_freemem.o set_Cparameters_to_default.o exact_solution_single_point.o exact_solution_all_points.o rhs_eval.o catch_cuda_error.o main.o
MoL_malloc_y_n_gfs.o: MoL_malloc_y_n_gfs.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c MoL_malloc_y_n_gfs.cu -o MoL_malloc_y_n_gfs.o

MoL_free_memory_y_n_gfs.o: MoL_free_memory_y_n_gfs.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c MoL_free_memory_y_n_gfs.cu -o MoL_free_memory_y_n_gfs.o

MoL_malloc_non_y_n_gfs.o: MoL_malloc_non_y_n_gfs.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c MoL_malloc_non_y_n_gfs.cu -o MoL_malloc_non_y_n_gfs.o

MoL_free_memory_non_y_n_gfs.o: MoL_free_memory_non_y_n_gfs.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c MoL_free_memory_non_y_n_gfs.cu -o MoL_free_memory_non_y_n_gfs.o

MoL_step_forward_in_time.o: MoL_step_forward_in_time.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c MoL_step_forward_in_time.cu -o MoL_step_forward_in_time.o

set_bcstruct.o: set_bcstruct.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c set_bcstruct.cu -o set_bcstruct.o

driver_bcstruct.o: driver_bcstruct.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c driver_bcstruct.cu -o driver_bcstruct.o

apply_bcs_curvilinear_inner_only.o: apply_bcs_curvilinear_inner_only.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c apply_bcs_curvilinear_inner_only.cu -o apply_bcs_curvilinear_inner_only.o

freemem_bcstruct.o: freemem_bcstruct.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c freemem_bcstruct.cu -o freemem_bcstruct.o

set_up__bc_gz_map_and_parity_condns.o: set_up__bc_gz_map_and_parity_condns.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c set_up__bc_gz_map_and_parity_condns.cu -o set_up__bc_gz_map_and_parity_condns.o

apply_bcs_curvilinear_radiation.o: apply_bcs_curvilinear_radiation.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c apply_bcs_curvilinear_radiation.cu -o apply_bcs_curvilinear_radiation.o

find_timestep.o: find_timestep.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c find_timestep.cu -o find_timestep.o

xx_to_Cart.o: xx_to_Cart.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c xx_to_Cart.cu -o xx_to_Cart.o

set_Nxx_dxx_invdx_params__and__xx.o: set_Nxx_dxx_invdx_params__and__xx.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c set_Nxx_dxx_invdx_params__and__xx.cu -o set_Nxx_dxx_invdx_params__and__xx.o

Cart_to_xx_and_nearest_i0i1i2.o: Cart_to_xx_and_nearest_i0i1i2.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c Cart_to_xx_and_nearest_i0i1i2.cu -o Cart_to_xx_and_nearest_i0i1i2.o

Cart_to_xx_and_nearest_i0i1i2_global_grid_center.o: Cart_to_xx_and_nearest_i0i1i2_global_grid_center.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c Cart_to_xx_and_nearest_i0i1i2_global_grid_center.cu -o Cart_to_xx_and_nearest_i0i1i2_global_grid_center.o

rfm_precompute_rfmstruct_malloc.o: rfm_precompute_rfmstruct_malloc.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c rfm_precompute_rfmstruct_malloc.cu -o rfm_precompute_rfmstruct_malloc.o

rfm_precompute_rfmstruct_define.o: rfm_precompute_rfmstruct_define.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c rfm_precompute_rfmstruct_define.cu -o rfm_precompute_rfmstruct_define.o

rfm_precompute_rfmstruct_freemem.o: rfm_precompute_rfmstruct_freemem.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c rfm_precompute_rfmstruct_freemem.cu -o rfm_precompute_rfmstruct_freemem.o

set_Cparameters_to_default.o: set_Cparameters_to_default.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c set_Cparameters_to_default.cu -o set_Cparameters_to_default.o

exact_solution_single_point.o: exact_solution_single_point.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c exact_solution_single_point.cu -o exact_solution_single_point.o

exact_solution_all_points.o: exact_solution_all_points.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c exact_solution_all_points.cu -o exact_solution_all_points.o

rhs_eval.o: rhs_eval.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c rhs_eval.cu -o rhs_eval.o

catch_cuda_error.o: catch_cuda_error.cu
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c catch_cuda_error.cu -o catch_cuda_error.o	

main.o: main.cu free_parameters.h
	$(CC) $(CFLAGS) $(INCLUDEDIRS)  -c main.cu -o main.o



ScalarWaveCurvilinear_Playground:   MoL_malloc_y_n_gfs.o MoL_free_memory_y_n_gfs.o MoL_malloc_non_y_n_gfs.o MoL_free_memory_non_y_n_gfs.o MoL_step_forward_in_time.o set_bcstruct.o driver_bcstruct.o apply_bcs_curvilinear_inner_only.o freemem_bcstruct.o set_up__bc_gz_map_and_parity_condns.o apply_bcs_curvilinear_radiation.o find_timestep.o xx_to_Cart.o set_Nxx_dxx_invdx_params__and__xx.o Cart_to_xx_and_nearest_i0i1i2.o Cart_to_xx_and_nearest_i0i1i2_global_grid_center.o rfm_precompute_rfmstruct_malloc.o rfm_precompute_rfmstruct_define.o rfm_precompute_rfmstruct_freemem.o set_Cparameters_to_default.o exact_solution_single_point.o exact_solution_all_points.o rhs_eval.o catch_cuda_error.o main.o
	$(CC)   MoL_malloc_y_n_gfs.o MoL_free_memory_y_n_gfs.o MoL_malloc_non_y_n_gfs.o MoL_free_memory_non_y_n_gfs.o MoL_step_forward_in_time.o set_bcstruct.o driver_bcstruct.o apply_bcs_curvilinear_inner_only.o freemem_bcstruct.o set_up__bc_gz_map_and_parity_condns.o apply_bcs_curvilinear_radiation.o find_timestep.o xx_to_Cart.o set_Nxx_dxx_invdx_params__and__xx.o Cart_to_xx_and_nearest_i0i1i2.o Cart_to_xx_and_nearest_i0i1i2_global_grid_center.o rfm_precompute_rfmstruct_malloc.o rfm_precompute_rfmstruct_define.o rfm_precompute_rfmstruct_freemem.o set_Cparameters_to_default.o exact_solution_single_point.o exact_solution_all_points.o rhs_eval.o catch_cuda_error.o main.o -o ScalarWaveCurvilinear_Playground -lm -lgomp

clean:
	rm -f *.o */*.o *~ */*~ ./#* *.txt *.dat *.avi *.png ScalarWaveCurvilinear_Playground
