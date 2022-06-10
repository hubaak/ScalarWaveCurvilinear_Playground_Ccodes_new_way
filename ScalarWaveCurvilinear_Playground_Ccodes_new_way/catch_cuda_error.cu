#include <stdio.h>
#include <string.h>
#include "./NRPy_basic_defines.h"
#include "./NRPy_function_prototypes.h"



__host__ void catch_cuda_error(char tag[]){
    cudaError_t error = cudaGetLastError();
    
    if(error != cudaSuccess)
    {
        char buf[1000] = "CUDA error tag ";
        strcat(buf, tag);
        strcat(buf, ": %s\n");
        // print the CUDA error message and exit
        printf(buf, cudaGetErrorString(error));
        exit(-1);
    }
}