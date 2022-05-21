
#include "definition.h"
#include <cuda_runtime.h> 



/* #if Q==32
    __global__ void GPU_VN( unsigned char * d_beta,  unsigned char * d_alpha,  unsigned char * d_alpha_t, unsigned char * d_alpha_t2,unsigned char * d_gamma, int iter);

    __global__ void GPU_CN( unsigned char * d_beta,  unsigned char * d_F,  unsigned char * d_B, int iter, unsigned char * d_h_nb);

    __global__ void GPU_FB_metrics(unsigned char * d_alpha, unsigned char * d_F, unsigned char * d_B, int iter, unsigned char * d_h_nb);

#else */
    __global__ void GPU_VN( uchar2 * d_beta_2,  uchar2 * d_alpha_2,  unsigned char * d_alpha_t, unsigned char * d_alpha_t2,unsigned char * d_gamma, int iter);

    __global__ void GPU_CN( uchar2 * d_beta_2,  uchar2 * d_F_2,  uchar2 * d_B_2, int iter);

    __global__ void GPU_FB_metrics(uchar2 * d_alpha_2, uchar2 * d_F_2, uchar2 * d_B_2, int iter);
//#endif

extern "C" int cuda_minmax(unsigned char* h_beta, unsigned char* ALPHAmn_, unsigned char* GAMMAn_, unsigned char* h_F,unsigned char* h_B,int* iteration);
